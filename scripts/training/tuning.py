# -*- coding: utf-8 -*-
# %%
"""old_rev_fixwandb_jp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mOZX6Qccjth7KoaZOY0Tx59yawtJ3DJ9

# Setup
"""
import os

TEST_RUN = os.environ.get("TEST_RUN", False)

datapath = os.environ.get("DATADIR", f"/content/drive/MyDrive/data/")
datapath = os.environ.get("DATADIR", f"/home/gdep-guest33/workarea/data/voice/")

modelpath = os.environ.get("AUDIOMODELDIR", f"/content/drive/MyDrive/data/models/")
modelpath = os.environ.get(
    "AUDIOMODELDIR", f"/home/gdep-guest33/workarea/models/voice/"
)

logpath = os.environ.get("AUDIOLOGDIR", f"/content/drive/MyDrive/data/logs/")
logpath = os.environ.get("AUDIOLOGDIR", f"/home/gdep-guest33/workarea/logs/voice/")


# from google.colab import drive
# drive.mount('/content/drive')
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
# pip install neptune-client mecab-python3 wandb
# pip install fastai -Uqq
#
# touch installed
# fi

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %load_ext tensorboard
# %autoreload 2
# import os 
# os.chdir("/content/itps_voice_recognition")

"""# Tensorboard

# Imports
"""
import csv
import os 
# os.chdir("/content/itps_voice_recognition/src/")

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
from itpsaudio.aug_transforms import (
    FrequencyMaskAugment,
    TimeMaskAugment,
    ToSpec,
    ToWave,
)
from itpsaudio.metrics import CER, WER
from dsets.dsets import ENGLISH_DATASETS, get_datasets, JAPANESE_DATASETS
from dsets.dset_config.dset_config import DatasetConfig
from dsets.helpers.helpers import apply_parallel, get_sampling_rates
from itpsaudio.transforms import *
from itpsaudio.core import *
from itpsaudio.modelling import SmoothCTCLoss, CTCLoss
from itpsaudio.tf_tokenizers import JPTransformersTokenizer, ENTransformersTokenizer

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
import wandb
from tqdm.auto import tqdm

tqdm.pandas()

"""# Data Preparation"""


@lru_cache(maxsize=None)
def get_audio_length(s):
    t, sr = torchaudio.load(s)
    return len(t[0]) / sr, sr


def prepare_df(df, audio_length=10, min_audio_length=2):
    df[["audio_length", "sr"]] = df["filename"].progress_apply(
        lambda x: pd.Series(get_audio_length(x))
    )
    print("Longest clip: ", df["audio_length"].max())
    df["audio_length"].plot.hist()
    plt.show()
    print("Length of datset before filtering:", df["audio_length"].sum() / 60 / 60)
    df = df[df["audio_length"] < audio_length].reset_index(drop=True)
    # df = df[df["audio_length"] > min_audio_length].reset_index(drop=True)
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
        return TensorAudio(self.samplers[x.sr](x), sr=x.sr)


@Transform
def prepare_for_tokenizer(x: str):
    return x.lower().replace(" ", "|")


"""# Training

## Metrics
"""


def get_metrics(tok):
    return [Perplexity(), CER(tok), WER(tok)]


"""## Dataloader"""


def get_dls(
    df: pd.DataFrame,
    augs: List[Union[Transform, None]],
    with_attentions: bool,
    bs=4,
    text_pad_val=0,
):
    splits = RandomSplitter(valid_pct=0.2)(df)
    tfms = TfmdLists(df, AudioBatchTransform(), splits=splits)
    train_text_lens = df.loc[splits[0], "audio_length"].to_list()
    val_text_lens = df.loc[splits[1], "audio_length"].to_list()
    srtd_dl = partial(SortedDL, res=train_text_lens)
    dl_kwargs = [{}, {"val_res": val_text_lens}]
    dls = tfms.dataloaders(
        bs=bs,
        after_item=[Resampler(df["sr"].unique()), prepare_for_tokenizer, tok] + augs,
        before_batch=[
            Pad_Audio_Batch(
                pad_idx_audio=0,
                pad_idx_text=text_pad_val,
                pad_first=True,
                seq_len=1,
                with_attention_masks=with_attentions,
            ),
            squeeze,
        ],
        shuffle=True,
        n_inp=1,
        dl_type=srtd_dl,
        dl_kwargs=dl_kwargs,
    )
    return dls


def construct_augs(params) -> List[Union[None, Transform]]:
    augs = []
    random_reverb = params.pop("RandomReverb")
    freq_mask = params.pop("FreqMask")
    time_mask = params.pop("TimeMask")
    if random_reverb:
        augs += [RandomReverbration(p=0.1)]
    if freq_mask or time_mask:
        augs.append(ToSpec())
        if freq_mask:
            augs.append(FrequencyMaskAugment(p=0.2))
        if time_mask:
            augs.append(TimeMaskAugment(p=0.2))
        augs.append(ToWave())
    return augs


def write_csv(fname, columns, data):
    with open(fname, "w") as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(columns)
        spamwriter.writerows(data)
    return fname


@patch
def after_epoch(self: WandbCallback):
    "Log validation loss and custom metrics & log prediction samples"
    # Correct any epoch rounding error and overwrite value
    self._wandb_epoch = round(self._wandb_epoch)
    if self.log_preds and self.log_preds_every_epoch:
        self.log_predictions()
    wandb.log({"epoch": self._wandb_epoch}, step=self._wandb_step)
    wandb.log(
        {
            "wandbcb" + n: s
            for n, s in zip(self.recorder.metric_names, self.recorder.log)
            if n not in ["train_loss", "epoch", "time"]
        },
        step=self._wandb_step,
    )


def log_itps_predictions(learn, dls, df):
    caption = "ITPS predictions"
    tfms = TfmdLists(df, AudioBatchTransform())
    dl = dls.test_dl(
        tfms,
        shuffle=False,
        val_res=df["audio_length"],
        after_item=dls.after_item,
        newline="",
        before_batch=dls.before_batch,
    )
    with torch.no_grad():
        logits, ys = learn.get_preds(dl=dl, with_input=False, with_decoded=False)
    acts = logits.map(lambda x: torch.argmax(x, dim=-1).detach())
    table_row = []
    csv_row = []
    for y, pred, true_y in tqdm(zip(ys, acts, df["text"].to_list()), total=len(ys)):
        dec_y = tok.decode(
            y, group_tokens=False, skip_special_tokens=False, skip_unk=True
        )
        dec_pred = tok.decode(
            pred, group_tokens=True, skip_special_tokens=False, skip_unk=True
        )
        table_row.append([true_y, dec_y, dec_pred])
        csv_row.append([true_y, dec_pred, dec_y])
    wandb.log(
        {caption: wandb.Table(columns=["true_y", "tok_y", "pred"], data=table_row)}
    )
    write_csv(
        Path(datapath)
        / "csv"
        / f"{LANG}"
        / "itps_predictions"
        / (wandb.run.name + ".csv"),
        columns=["true_y", "pred", "tok_y"],
        data=csv_row,
    )
    return dec_y, dec_pred, logits, dl


def log_predictions(learn, dls, cer, thresh=0.2):
    caption = "Predictions"
    tfms = TfmdLists(df, AudioBatchTransform())
    dl = dls.test_dl(
        tfms,
        shuffle=False,
        val_res=df["audio_length"],
        after_item=dls.after_item,
        newline="",
        before_batch=dls.before_batch,
    )
    with torch.no_grad():
        logits, ys = learn.get_preds(dl=dl, with_input=False, with_decoded=False)
    acts = logits.map(lambda x: torch.argmax(x, dim=-1).detach())
    table_row = []
    csv_row = []
    for y, pred, true_y in tqdm(zip(ys, acts, df["text"].to_list()), total=len(ys)):
        dec_y = tok.decode(
            y, group_tokens=False, skip_special_tokens=False, skip_unk=True
        )
        dec_pred = tok.decode(
            pred, group_tokens=True, skip_special_tokens=False, skip_unk=True
        )
        table_row.append([true_y, dec_y, dec_pred])
        csv_row.append([true_y, dec_pred, dec_y])
    wandb.log(
        {caption: wandb.Table(columns=["true_y", "tok_y", "pred"], data=table_row)}
    )
    if cer < thresh:
        write_csv(
            Path(datapath) / "csv" / f"{LANG}" / (wandb.run.name + ".csv"),
            columns=["true_y", "pred", "tok_y"],
            data=csv_row,
        )
    return dec_y, dec_pred, logits


"""## Model"""


class MetricsToWandb(Callback):
    def after_epoch(self):
        "Log validation loss and custom metrics & log prediction samples"
        wandb.log(
            {
                n: s
                for n, s in zip(self.recorder.metric_names, self.recorder.log)
                if n not in ["epoch", "time"]
            }
        )


class DropPreds(Callback):
    def after_pred(self):
        self.learn.pred = self.pred.logits


def get_model(
    arch,
    tok,
    with_attentions=False,
    attention_dropout=0,
    hidden_dropout=0,
    feat_proj_dropout=0,
    layerdrop=0,
    mask_time_prob=0,
    mask_feature_prob=0,
    **kwargs,
):

    if with_attentions:
        model = AutoModelForCTC.from_pretrained(
            arch,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            feat_proj_dropout=feat_proj_dropout,
            layerdrop=layerdrop,
            pad_token_id=0,
            vocab_size=len(tok.tokenizer),
            ctc_loss_reduction="sum",
            ctc_zero_infinity=True,
        )
        return model
    else:
        model = AutoModelForCTC.from_pretrained(
            arch,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            feat_proj_dropout=feat_proj_dropout,
            mask_time_prob=mask_time_prob,
            mask_feature_prob=mask_feature_prob,
            layerdrop=layerdrop,
            ctc_zero_infinity=True,
            pad_token_id=tok.tokenizer.pad_token_id,
            vocab_size=len(tok.tokenizer),
        )
        return model


def get_logging_cbs(framework, params=None, **kwargs):
    if framework.lower() == "wandb":
        wandb.init(project="itps-gpu-real", config=params)
        log_cbs = [WandbCallback(log="all", log_preds=False, log_model=False, **kwargs)]
    elif framework.lower() == "neptune":
        neptune.init("jjs/itps-language-model")
        log_cbs = [
            NeptuneCallback(
                log_model_weights=False, keep_experiment_running=False, **kwargs
            ),
            NeptuneSaveModel(1),
        ]
    else:
        log_cbs = []
    return log_cbs


def get_learner(
    dls, model, loss_func, modelpath, cbs, log_cbs, metrics, with_attentions
):
    if isinstance(loss_func, str):
        if loss_func == "transformers_ctc":
            learn = TransformersLearner(
                dls=dls,
                model=model,
                loss_func=noop,
                metrics=metrics,
                model_dir=modelpath,
                cbs=cbs + log_cbs,
            )
        else:
            assert False, "Unknown loss function"
    else:
        learn = TransformersLearnerOwnLoss(
            dls=dls,
            model=model,
            loss_func=loss_func,
            metrics=metrics,
            model_dir=modelpath,
            cbs=cbs + log_cbs,
            with_attentions=with_attentions,
        )
    return learn


"""# Optuna run"""

LOGGING_FRAMEWORK = "wandb"

"""# Trial Suggestions"""


def trial_suggestions(trial):
    with_attentions = trial.suggest_categorical("with_attentions", [True, False])
    loss_func = trial.suggest_categorical(
        "loss_func", ["smooth_ctc_sum", "transformers_ctc", "ctc_sum", "ctc_mean"]
    )
    bs = trial.suggest_categorical("bs", [8])
    # This is weird but I wanna have the augs in the trial object
    # Achitecture hyperparams
    big_archlist = [
        "facebook/hubert-xlarge-ll60k",
        "facebook/wav2vec2-xls-r-1b",
        #  "facebook/wav2vec2-xls-r-2b"
    ]
    smaller_archlist = [
        "facebook/hubert-large-ll60k",
        "facebook/wav2vec2-xls-r-300m",
    ]
    arch = trial.suggest_categorical("arch", smaller_archlist)

    feat_proj_dropout = trial.suggest_float("feat_proj_dropout", low=0.03, high=0.2)
    hidden_dropout = trial.suggest_float("hidden_dropout", low=0.03, high=0.2)
    attention_dropout = trial.suggest_float("attention_dropout", low=0.03, high=0.2)
    layerdrop = trial.suggest_float("layerdrop", low=0.03, high=0.2)
    mask_time_prob = trial.suggest_float("mask_time_prob", low=0.03, high=0.3)
    mask_feature_prob = trial.suggest_float("mask_feature_prob", low=0.03, high=0.3)

    # Augmentations
    random_reverb = trial.suggest_categorical("RandomReverb", [True, False])
    freq_mask = trial.suggest_categorical("FreqMask", [False])
    time_mask = trial.suggest_categorical("TimeMask", [False])
    # lr
    lr = trial.suggest_float("lr", low=3e-5, high=1e-3)
    if loss_func != "transformers_ctc":
        lr_finder = trial.suggest_categorical("lr_finder", [True, False])

    # sched = trial.suggest_categorical("schedule", [
    #   "fit_one_cycle", "exponential",# "ramp_quarter_pipe"
    # ])

    # def after_epoch(self):
    #     "Log validation loss and custom metrics & log prediction samples"
    #     # Correct any epoch rounding error and overwrite value
    #     self._wandb_epoch = round(self._wandb_epoch)
    #     if self.log_preds and self.log_preds_every_epoch:
    #         self.log_predictions()
    #     wandb.log({'epoch': self._wandb_epoch}, step=self._wandb_step)
    #     wandb.log({n:s for n,s in zip(self.recorder.metric_names, self.recorder.log) if n not in ['train_loss', 'epoch', 'time']}, step=self._wandb_step)


def get_loss_func(loss_func, num_classes, **kwargs):
    if loss_func == "ctc_mean":
        return CTCLoss(num_classes, reduction="mean", **kwargs)
    elif loss_func == "ctc_sum":
        return CTCLoss(num_classes, reduction="sum", **kwargs)
    elif loss_func == "smooth_ctc_sum":
        return SmoothCTCLoss(num_classes, ctc_red="sum", **kwargs)
    elif loss_func == "smooth_ctc_mean":
        return SmoothCTCLoss(num_classes, ctc_red="mean", **kwargs)
    elif loss_func == "transformers_ctc":
        return "transformers_ctc"


def save_json(fname, obj):
    with open(fname, "w") as f:
        json.dump(obj, f)


def run(input_pars, modelpath, logpath):
    params = input_pars.copy()
    log_cbs = get_logging_cbs(framework=LOGGING_FRAMEWORK, params=input_pars)

    with_attentions = params.pop("with_attentions")
    loss_func_name = params.pop("loss_func")
    arch = params.pop("arch")
    lr = params.pop("lr")
    bs = params.pop("bs")
    augs = construct_augs(params)

    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
    date = datetime.datetime.now().strftime("%y%m%d%H%MD")
    log_base_path = Path(logpath)

    archname = f"{arch.replace('/', '_')}_{timestamp}"

    model_out_path = f"{modelpath / wandb.run.name}"
    logdir = Path(f"{log_base_path / wandb.run.name}")

    model = get_model(arch, tok, with_attentions=with_attentions, **params)
    loss_func = get_loss_func(loss_func_name, len(tok.tokenizer))

    if isinstance(loss_func, str):
        if loss_func == "transformers_ctc":
            dls = get_dls(df, augs, with_attentions, bs=bs, text_pad_val=-100)
    else:
        dls = get_dls(df, augs, with_attentions, bs=bs, text_pad_val=-100)

    fit_cbs = [
        SeePreds(
            arch,
            tok,
            n_iters=len(dls.train) // 5,
            log_dir=logdir,
            neptune=(LOGGING_FRAMEWORK.lower() == "neptune"),
        ),
        TensorBoardCallback(log_dir=logdir, trace_model=False, log_preds=False),
        EarlyStoppingCallback(
            comp=np.less, monitor=MONITOR, patience=2, min_delta=0.001
        ),
        MetricsToWandb(),
    ]

    # sched = params.pop("schedule")
    lr_max, lr_min = lr, lr / 10
    schedule = ParamScheduler({"lr": SchedExp(start=lr_max, end=lr_min)})
    fit_cbs.append(schedule)
    # if sched == "exponential":
    #     lr_min, lr_max = lr, lr/10
    #     schedule = SchedExp(start=lr_max, end=lr_min)
    #     fit_cbs.append(schedule)
    # elif sched == "ramp_quarter_pipe":
    #       lr_min, lr_max = lr, lr/10
    #       schedule = combine_scheds([0.3, 0.2, 0.5], [SchedLin(lr_min, lr_max), SchedNo(lr_max, lr_max), SchedCos(lr_max, lr_min)])
    #       fit_cbs.append(schedule)

    model.freeze_feature_extractor()

    if LOGGING_FRAMEWORK.lower() == "wandb":
        wandb.log(input_pars)
        wandb.run.summary["epoch"] = 0
        wandb.config.update({"_ENV": dict(os.environ)})

    x = dls.show_batch(tok=tok, skip_special_tokens=True)
    wandb.log({"input_examples": plt})
    b = dls.one_batch()
    for x, y in zip(b[0], b[-1]):
        wandb.log(
            {
                "input_audio": wandb.Audio(
                    x.detach().cpu().numpy(),
                    caption=tok.decode(y, group_tokens=False),
                    sample_rate=16000,
                )
            }
        )

    learn = get_learner(
        dls=dls,
        model=model,
        loss_func=loss_func,
        metrics=get_metrics(tok),
        modelpath=model_out_path,
        with_attentions=with_attentions,
        cbs=[
            DropPreds(),
            SaveModelCallback(
                comp=np.less,
                monitor=MONITOR,
                min_delta=0.001,
                fname=arch.replace("/", "_"),
            ),
        ],
        log_cbs=log_cbs,
    )
    if TEST_RUN:
        # if sched == "fit_one_cycle":
        #   learn.fit_one_cycle(1, lr_max=lr, cbs=fit_cbs)
        # else:
        learn.fit(1, lr=lr, cbs=fit_cbs)
    else:
        # if sched == "fit_one_cycle":
        #   learn.fit_one_cycle(100, lr_max=lr, cbs=fit_cbs)
        # else:
        learn.fit(100, lr=lr, cbs=fit_cbs)

    valid_loss, perplexity, wer, cer = learn.validate()

    valid_loss, perplexity, wer, cer
    log_predictions(learn, dls, cer)

    _, itps_data = get_datasets(
        [DatasetConfig(name="itps", lang=LANG, split="train")], base="~/.fastdownload"
    )
    if not os.path.exists(Path(datapath) / f"itps_data_{LANG}.pkl"):
        itps_df = prepare_df(itps_data, audio_length=AUDIO_LENGTH)
    else:
        itps_df = pd.read_pickle(datapath / f"itps_data_{LANG}.pkl")
    _, _, _, itps_dl = log_itps_predictions(learn, dls, itps_df)
    print("Got all the predictions for itps")

    valid_loss, perplexity, wer, cer = learn.validate(dl=itps_dl)
    print(f"Got evaluation for itps WER: {wer}, CER: {cer}")
    wandb.log(
        {
            "itps_valid_loss": valid_loss,
            "itps_perplexity": perplexity,
            "wer": wer,
            "cer": cer,
        }
    )

    if cer < 0.2:
        learn.model.save_pretrained(model_out_path)
        save_json(Path(model_out_path) / "vocab.json", tok.tokenizer.get_vocab())

    wandb.finish()
    return cer


LANG = os.environ.get("TRAIN_LANG", "jp")
if LANG == "jp":
    datasets = JAPANESE_DATASETS[1:]
if LANG == "en":
    datasets = [
        # DatasetConfig(name='itps', split='train', lang='en', kind=None),
        DatasetConfig(name="librispeech", split="dev", lang=None, kind="clean"),
        DatasetConfig(name="ljl", split="train", lang=None, kind=None),
        DatasetConfig(name="nict_spreds", split="train", lang="en", kind=None),
    ]
MONITOR = "cer"
NUM_EPOCHS = 20
AUDIO_LENGTH = 10

p, df = get_datasets(datasets, base="~/.datasets")

cv = load_dataset("common_voice", f"{'ja' if LANG == 'jp' else 'en'}")
paths = pd.Series(cv["train"]["path"])
sentences = pd.Series(cv["train"]["sentence"])
cv = pd.DataFrame({"filename": paths, "text": sentences})

DSET_NAMES = "-".join([f"{LANG}"] + [d.name[:4] + "-" + d.split[:4] for d in datasets])

if LANG == "en":
    frac = 0.01
    cv = cv.sample(frac=frac)
    DSET_NAMES += f"{frac:.2f}".replace(".", "_")
    print(f"{frac:.2f}")

df = pd.concat([cv, df], ignore_index=True).reset_index(drop=True)
DSET_NAMES += "-cv"


FLEURS = False
if LANG == "jp" and FLEURS:
    fleurs_asr = load_dataset("google/xtreme_s", "fleurs.ja_jp")
    fl_tr = pd.DataFrame(
        {
            "filename": pd.Series(fleurs_asr["train"]["path"]),
            "text": pd.Series(fleurs_asr["train"]["raw_transcription"]),
        }
    )
    fl_val = pd.DataFrame(
        {
            "filename": pd.Series(fleurs_asr["validation"]["path"]),
            "text": pd.Series(fleurs_asr["validation"]["raw_transcription"]),
        }
    )
    fl_test = pd.DataFrame(
        {
            "filename": pd.Series(fleurs_asr["test"]["path"]),
            "text": pd.Series(fleurs_asr["test"]["raw_transcription"]),
        }
    )
    fl = pd.concat([fl_tr, fl_val, fl_test], ignore_index=True)
    df = pd.concat([df, fl], ignore_index=True).reset_index(drop=True)
    DSET_NAMES += "-fleurs"


# %%capture
# %%bash
# if [ ! -f installed_mecab ]; then
# apt-get -q -y install mecab file libmecab-dev mecab-ipadic-utf8 git curl
# git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
# echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n  -y
# ln -s /etc/mecabrc /usr/local/etc/mecabrc
# touch installed_mecab
# fi

"""# Actual Run"""

dfpath = (
    Path().home() / ".fastdownload" / "preprocessed" / f"{LANG}_df_{DSET_NAMES}.csv"
)
# if not os.path.exists(dfpath):
#   print('Preparing df')
df = prepare_df(df, audio_length=AUDIO_LENGTH)
df.to_pickle(dfpath)
# else:
#   print('reading df')
#   df = pd.read_pickle(dfpath)
#   if TEST_RUN:
#     df = df.iloc[:100]

print(DSET_NAMES)

if LANG == "jp":
    vocab = JPTransformersTokenizer.create_vocab("vocab.json")
    tok = JPTransformersTokenizer(
        tok=Wav2Vec2CTCTokenizer(
            vocab,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
    )
else:
    vocab = ENTransformersTokenizer.create_vocab("vocab.json")
    tok = ENTransformersTokenizer(
        tok=Wav2Vec2CTCTokenizer(
            vocab,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
    )

modelpath = Path(modelpath) / f"audio_{LANG}"
logpath = Path(logpath) / f"audio_{LANG}"

storage_fname = f"opt_study_{LANG}"
storage = f"{modelpath / storage_fname}"

STUDY_NAME = f"{LANG}_study"


def objective(trial):
    augs = trial_suggestions(trial)
    cer = run(trial.params, modelpath, logpath)
    return cer


study = optuna.create_study(
    "sqlite:///{}.db".format(storage),
    direction="minimize",
    study_name=STUDY_NAME,
    load_if_exists=True,
)


# Snippet for trying hyperparamters
# params = {'with_attentions': False,
# 'loss_func': 'ctc_sum',
# 'bs': 8,
# 'arch': 'facebook/hubert-large-ll60k',
# 'freeze_feat': True,
# 'feat_proj_dropout': 0.10,
# 'hidden_dropout': 0.10,
# 'attention_dropout': 0.10,
# 'layerdrop': 0.10,
# 'mask_time_prob': 0.10,
# 'mask_feature_prob': 0.10,
# 'RandomReverb': True,
# 'FreqMask': False,
# 'TimeMask': False,
# 'lr': 1e-4}
# study.enqueue_trial(params)

all_studies = optuna.get_all_study_summaries("sqlite:///{}.db".format(storage))
# %%
study.optimize(objective, n_trials=10)
trial = study.best_trial
# %%
def quick_get_run(input_pars, modelpath, logpath):
    params = input_pars.copy()
    with_attentions = params.pop("with_attentions")
    loss_func_name = params.pop("loss_func")
    freeze_feat = params.pop("freeze_feat")
    arch = params.pop("arch")
    lr = params.pop("lr")
    bs = params.pop("bs")
    augs = construct_augs(params)
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
    date = datetime.datetime.now().strftime("%y%m%d%H%MD")
    log_base_path = Path(logpath)
    archname = f"{arch.replace('/', '_')}_{timestamp}"

    model_out_path = f"{modelpath / archname}"
    logdir = Path(f"{log_base_path / archname}")

    fit_cbs = [
        ParamScheduler({"lr": SchedExp(start=lr, end=lr / 10)}),
        SeePreds(
            arch,
            tok,
            n_iters=500,
            log_dir=logdir,
            neptune=(LOGGING_FRAMEWORK.lower() == "neptune"),
        ),
        TensorBoardCallback(log_dir=logdir, trace_model=False, log_preds=False),
        EarlyStoppingCallback(
            comp=np.less, monitor=MONITOR, min_delta=0.01, patience=2
        ),
    ]

    model = get_model(arch, tok, with_attentions=with_attentions, **params)
    loss_func = get_loss_func(loss_func_name, len(tok.tokenizer))

    if loss_func == "transformers_ctc":
        dls = get_dls(df, augs, with_attentions, bs=bs, text_pad_val=-100)
    else:
        dls = get_dls(df, augs, with_attentions, bs=bs, text_pad_val=-100)
    dls.show_batch(tok=tok, skip_special_tokens=True)
    plt.show()
    model.freeze_feature_extractor()
    log_cbs = get_logging_cbs(framework=LOGGING_FRAMEWORK)
    if LOGGING_FRAMEWORK.lower() == "wandb":
        wandb.log(input_pars)
        wandb.run.summary["epoch"] = 0
    learn = get_learner(
        dls=dls,
        model=model,
        loss_func=loss_func,
        metrics=get_metrics(tok),
        modelpath=model_out_path,
        with_attentions=with_attentions,
        cbs=[DropPreds()],
        log_cbs=log_cbs,
    )
    return learn, dls

