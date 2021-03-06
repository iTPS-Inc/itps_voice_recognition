# -*- coding: utf-8 -*-
"""english_own_loss.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RsWucm4bDYzVcPfvGRt2LW9TCTzDdoBi

# Setup
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %%bash 
# cd /content/
# if [ -d itps_voice_recognition ]; then
#   cd itps_voice_recognition
#   git pull -r 
#   cd ..
# else 
#   git clone https://github.com/iTPS-Inc/itps_voice_recognition.git
# fi
# 
# if [ ! -f installed ]; then 
# 
# pip install boto3 datasets==1.13.3 transformers==4.11.3 librosa jiwer sentencepiece japanize_matplotlib optuna
# pip install neptune-client mecab-python3
# pip install fastai -Uqq
# 
# touch installed
# fi

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %load_ext tensorboard
# %autoreload 2
import os 
os.chdir("/content/itps_voice_recognition")
from dsets.dsets import ENGLISH_DATASETS, get_datasets, JAPANESE_DATASETS
from dsets.dset_config.dset_config import DatasetConfig

from dsets.helpers.helpers import apply_parallel, get_sampling_rates
from itpsaudio.transforms import * 
from itpsaudio.core import * 
from itpsaudio.modelling import SmoothCTCLoss, CTCLoss

"""# Tensorboard"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir '/content/drive/MyDrive/data/logs/audio_en'

"""# Imports """

from fastai.data.all import * 
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForMaskedLM, AutoTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, AutoModelForCTC
from transformers import Wav2Vec2CTCTokenizer
from datasets import load_dataset
from fastai.text.all import * 
from fastai.vision.all import *
from fastai.callback.tensorboard import *
from fastai.callback.neptune import *
import torchaudio.transforms as T

from itpsaudio.callbacks import SeePreds, NeptuneSaveModel
from itpsaudio.aug_transforms import AddNoise, RandomReverbration
from itpsaudio.callbacks import MixedPrecisionTransformers
from itpsaudio.aug_transforms import FrequencyMaskAugment, TimeMaskAugment, ToSpec, ToWave
from itpsaudio.metrics import CER, WER

from functools import lru_cache
import torchaudio
import pandas
import optuna
import MeCab
from pathlib import Path
from IPython.display import Audio, display
import japanize_matplotlib
import jiwer
import json
import pickle
from datasets import load_metric
import neptune as neptune
import datetime 
from functools import lru_cache
from collections import Counter

"""# Data Preparation"""

def make_vocab(outpath, text_col):
  c = Counter("|".join(text_col.str.lower().to_list()))
  allowed_letters = string.ascii_lowercase+" .?!|-,\"'"
  print("Allowing letters: ")
  print(allowed_letters)
  vocab = {}
  special_toks = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
  vocab = special_toks + [i for i in allowed_letters]
  vocab = {v: i for i, v in enumerate(vocab)}
  with open(outpath, "w") as f:
    json.dump(vocab, f)
  with open(outpath, "r") as f: _ = json.load(f)
  return outpath

@lru_cache(maxsize=None)
def get_audio_length(s):
    t, sr = torchaudio.load(s)
    return len(t[0]) / sr, sr
def prepare_df(df, audio_length=10):
    df[["audio_length", "sr"]] = df["filename"].apply(
        lambda x: pd.Series(get_audio_length(x))
    )
    print("Longest clip: ", df["audio_length"].max())
    df["audio_length"].plot.hist()
    plt.show()
    print("Length of datset before filtering:", df["audio_length"].sum() / 60 / 60)
    df = df[df["audio_length"] < audio_length].reset_index(drop=True)
    df = df[~df["text"].isna()].reset_index(drop=True)
    df["text"] = df["text"].str.lower()
    print("Length of dataset after filtering: ", df["audio_length"].sum() / 60 / 60)
    df["audio_length"].plot.hist()
    return df

class Resampler(Transform):
    samplers = {16000: noop}
    def __init__(self, unique_srs):
      for sr in unique_srs:
        self.samplers[sr] = T.Resample(sr, 16000)

    def encodes(self, x: TensorAudio):
        return self.samplers[x.sr](x)

@Transform
def prepare_for_tokenizer(x: str):
    return x.lower().replace(" ", "|")

"""# Training

## Metrics
"""

def get_metrics(tok):
    return [Perplexity(), CER(tok), WER(tok)]

"""## Dataloader"""

def get_dls(df: pd.DataFrame, augs: List[Union[Transform, None]]):
    splits = RandomSplitter(valid_pct=0.2)(df)
    tfms = TfmdLists(df, AudioBatchTransform(), splits=splits)
    train_text_lens = df.loc[splits[0], "audio_length"].to_list()
    val_text_lens = df.loc[splits[1], "audio_length"].to_list()
    srtd_dl = partial(SortedDL, res=train_text_lens)
    dl_kwargs = [{}, {"val_res": val_text_lens}]
    dls = tfms.dataloaders(
        bs=4,
        after_item=[Resampler(df["sr"].unique()),
                    prepare_for_tokenizer, tok] + augs,
        before_batch=[
            Pad_Audio_Batch(
                pad_idx_audio=0,
                pad_idx_text=0,
                pad_first=True,
                seq_len=1,
                with_attention_masks=True,
            ),
            squeeze,
        ],
        shuffle=True,
        n_inp=1,
        dl_type=srtd_dl,
        dl_kwargs=dl_kwargs
    )
    return dls

def construct_augs(params) -> List[Union[None , Transform]]:
  augs = []
  random_reverb = params.pop("RandomReverb")
  freq_mask = params.pop("FreqMask")
  time_mask = params.pop("TimeMask")
  if random_reverb:
    augs += [RandomReverbration(p=0.1)]
  if freq_mask or time_mask:
    augs.append(ToSpec())
    if freq_mask: augs.append(FrequencyMaskAugment(p=0.2))
    if time_mask: augs.append(TimeMaskAugment(p=0.2))
    augs.append(ToWave())
  return augs

"""## Model"""

def get_model(
    arch,
    tok,
    attention_dropout=0,
    hidden_dropout=0,
    feat_proj_dropout=0,
    layerdrop=0,
):
    model = AutoModelForCTC.from_pretrained(
        arch,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        feat_proj_dropout=feat_proj_dropout,
        layerdrop=layerdrop,
        pad_token_id=0,
        vocab_size=len(tok.tokenizer),
    )
    return model


def get_learner(dls, model, loss_func, modelpath, cbs, metrics):
    learn = TransformersLearnerOwnLoss(
        dls=dls,
        model=model,
        loss_func=loss_func,
        metrics=metrics,
        model_dir=modelpath,
        cbs=cbs,
    )
    return learn

"""# Optuna run"""

NEPTUNE=False
if NEPTUNE:
    run = neptune.init("jjs/itps-language-model")
    log_cbs.append([
        NeptuneCallback(log_model_weights=False, keep_experiment_running=False),
        NeptuneSaveModel(1),
    ])

def trial_suggestions(trial):
    loss_func = trial.suggest_categorical( "loss_func", [ "smooth_ctc", "ctc"])
    # This is weird but I wanna have the augs in the trial object
    # Achitecture hyperparams
    archlist = [ "facebook/wav2vec2-base", "facebook/wav2vec2-xls-r-300m"]
    arch = trial.suggest_categorical("arch", archlist) 
    freeze_feat = trial.suggest_categorical("freeze_feat",    [True])

    feat_proj_dropout=trial.suggest_float("feat_proj_dropout", low=0, high=0.3)
    hidden_dropout=trial.suggest_float("hidden_dropout",       low=0, high=0.3)
    attention_dropout=trial.suggest_float("attention_dropout", low=0, high=0.3)
    layerdrop=trial.suggest_float("layerdrop",                 low=0, high=0.3)
    
    # Augmentations
    random_reverb = trial.suggest_categorical("RandomReverb", [True, False])
    freq_mask     = trial.suggest_categorical("FreqMask",     [True, False])
    time_mask     = trial.suggest_categorical("TimeMask",     [True, False])
    # lr
    lr = trial.suggest_float("lr", low=1e-5, high=1e-3)

def get_loss_func(loss_func, num_classes,reduction="mean", **kwargs):
  if loss_func == "ctc":
    return CTCLoss(num_classes, reduction="mean", **kwargs)
  elif loss_func == "smooth_ctc":
    return SmoothCTCLoss(num_classes,reduction="mean", **kwargs)

def run(params, modelpath, logpath):
  monitor="perplexity"
  params = params.copy()
  loss_func_name = params.pop("loss_func")
  freeze_feat = params.pop("freeze_feat")
  arch = params.pop("arch")
  lr = params.pop("lr")
  augs = construct_augs(params) 

  timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
  date=datetime.datetime.now().strftime("%y%m%d%H%MD")
  log_base_path = Path(logpath)

  archname = f"{arch.replace('/', '_')}_{timestamp}"
  model_out_path = f"{modelpath / archname}"
  logdir = Path(f"{log_base_path / archname}")

  fit_cbs=[
    ParamScheduler({"lr": SchedExp(start=lr, end=lr/10)}),
    TensorBoardCallback(log_dir=logdir, trace_model=False, log_preds=False),
    SeePreds(arch, tok, n_iters=500, log_dir=logdir, neptune=NEPTUNE),
    SaveModelCallback(fname=arch.replace("/", "_")),
    EarlyStoppingCallback(monitor=monitor, patience=2),
  ]

  model = get_model(arch, tok, **params)
  loss_func = get_loss_func(loss_func_name, len(tok.tokenizer), reduction="mean")

  if freeze_feat: model.freeze_feature_extractor()

  dls = get_dls(df, augs)
  learn = get_learner(
      dls=dls,
      model=model,
      loss_func=loss_func,
      metrics=get_metrics(tok),
      modelpath=model_out_path,
      cbs=[],
  )
  learn.fit(100, lr=lr, cbs=fit_cbs)
  learn.unfreeze()
  learn.fit(100, lr=lr/2, cbs=fit_cbs)
  return learn.validate()

datasets = [
#  DatasetConfig(name='itps', split='train', lang='en', kind=None),
 DatasetConfig(name='librispeech', split='train', lang=None, kind='clean'),
 DatasetConfig(name='ljl', split='train', lang=None, kind=None),
 DatasetConfig(name='nict_spreds', split='train', lang='en', kind=None)
]

dset_lengths = None
TEST_RUN=False
LAST_EPOCH=0
NUM_EPOCHS=20
AUDIO_LENGTH=10
DSET_NAMES = "-".join(["cven"] + [d.name[:4] for d in datasets])
print(DSET_NAMES)

p, df = get_datasets(datasets)

"""# Actual Run"""

if not os.path.exists("/content/df.pkl"):
    df = prepare_df(df, audio_length=AUDIO_LENGTH)
    df.to_pickle("/content/df.pkl")
else:
    df = pd.read_pickle("/content/df.pkl")

tok = ENTransformersTokenizer(
    tok=Wav2Vec2CTCTokenizer(
        make_vocab("vocab.json", df["text"]),
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
)

LANG = "en"
modelpath=Path(f"/content/drive/MyDrive/data/models/audio_{LANG}/")
logpath = Path(f"/content/drive/MyDrive/data/logs/audio_{LANG}/")

storage=f"sqlite:///{modelpath / 'testing_optuna_study.db' }"
all_studies = optuna.get_all_study_summaries(storage)

tok.tokenizer.get_vocab()

def objective(trial):
  augs = trial_suggestions(trial)
  # log_trial(params)
  valid_loss, perp, wer, cer = run(trial, modelpath, logpath)
  # log_trial(params)

  return perp

study = optuna.create_study(storage,
                            direction="minimize",
                            study_name="test_2", load_if_exists=True)
study.optimize(objective, n_trials=1)
trial = study.best_trial

trial = study.trials[0]

trial.params

def get_dls(df: pd.DataFrame, augs: List[Union[Transform, None]]):
    splits = RandomSplitter(valid_pct=0.2)(df)
    tfms = TfmdLists(df, AudioBatchTransform(), splits=splits)
    train_text_lens = df.loc[splits[0], "audio_length"].to_list()
    val_text_lens = df.loc[splits[1], "audio_length"].to_list()
    srtd_dl = partial(SortedDL, res=train_text_lens)
    dl_kwargs = [{}, {"val_res": val_text_lens}]
    dls = tfms.dataloaders(
        bs=4,
        after_item=[Resampler(df["sr"].unique()),
                    prepare_for_tokenizer, tok] + augs,
        before_batch=[
            Pad_Audio_Batch(
                pad_idx_audio=0,
                pad_idx_text=0,
                pad_first=True,
                seq_len=1,
                with_attention_masks=True,
            ),
            squeeze,
        ],
        shuffle=True,
        n_inp=1,
        dl_type=srtd_dl,
        dl_kwargs=dl_kwargs
    )
    return dls

df["text"]

run(trial, )

"""# Test running"""

if TEST_RUN:
    learn.fit_one_cycle(1, lr_max=1e-4)
else:
    sched = {"lr": SchedExp(1e-4, 1e-5)}
    fit_cbs = [
        SeePreds(
            pretrained_model_name, tok, n_iters=500, log_dir=logdir, neptune=NEPTUNE
        ),
        ParamScheduler(sched),
        SaveModelCallback(fname=pretrained_model_name.replace("/", "_")),
    ]
    learn.fit(NUM_EPOCHS, cbs=fit_cbs)

    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)
    learn.save(Path(model_out_path) / "learn_cb", with_opt=True)

if NEPTUNE:
    neptune.stop()

if not TEST_RUN:
    learn.model.save_pretrained(model_out_path)
    learn.save(Path(model_out_path) / "learner.falearner")
    with open(Path(model_out_path) / "vocab.json", "w") as f:
        json.dump(tok.tokenizer.get_vocab(), f)
    torch.save(learn.model, Path(model_out_path) / "raw_mod_exp.pth")

test_datasets = [
    DatasetConfig(name="itps", lang="jp", split="test"),
    DatasetConfig(name="jsut", split="test", lang=None, kind=None),
    DatasetConfig(name="nict_spreds", split="test", lang="jp", kind=None),
]

tp, tdf = get_datasets(test_datasets)

tdf["audio_length"] = apply_parallel(tdf["filename"], get_audio_length, 16)
tdf = tdf[tdf["audio_length"] < 15].reset_index(drop=True)
tdf = tdf[~tdf["text"].isna()].reset_index(drop=True)

abt = AudioBatchTransform()
t_tfms = TfmdLists(tdf, abt)

if TEST_RUN:
    t_tfms = TfmdLists(tdf.iloc[:100], abt)
else:
    t_tfms = TfmdLists(tdf, abt)

t_dl = dls.new(t_tfms)

t_dl.show_batch(tok=tok)


def get_preds(xs):
    preds = learn.model(xs)
    pred_logits = preds.logits
    pred_ids = TensorText(np.argmax(pred_logits.detach().cpu().numpy(), axis=-1))
    pred_str = tok.batch_decode(pred_ids)
    return pred_str


import pprint

for xs, _, _, y in iter(t_dl):
    print(wer(learn.model(xs), y))
    print(cer(learn.model(xs), y))
    pprint.pprint(dict(enumerate(list(zip(get_preds(xs), tok.batch_decode(y))))))
    break

comp = [(get_preds(xs), tok.batch_decode(y)) for xs, y in iter(t_dl)]

learn.model

for i, (x_pair, y_pair) in enumerate(comp):

    print("Pred: ", x_pair[0])
    print("Targ: ", y_pair[0])
    print("Pred: ", x_pair[1])
    print("Targ: ", y_pair[1])
    if (i + 1 % 10) == 0:
        break

with open("/content/drive/MyDrive/data/models/audio_jp/good_jp_mod/tok.pkl", "rb") as f:
    t = pickle.load(f)

