import torchaudio.backend.sox_io_backend as torchaudio_io
from fastai.data.all import (
    ItemTransform,
    TitledStr,
    Transform,
    store_attr,
    tensor,
    retain_type,
)
from fastai.text.all import TensorText, pad_chunk
import torch

from itpsaudio.core import AudioPair, TensorAudio, TensorAttention


@Transform
def extract_first(s: TensorAudio):
    if s.shape[0] == 1:
        return s[0]
    else:
        return s


@Transform
def capitalize(s: str):
    return s.upper()


@Transform
def squeeze(s):
    return s.squeeze()


class TargetProcessor(Transform):
    def __init__(self, proc):
        self.proc = proc

    def encodes(self, y):
        with self.proc.as_target_processor():
            return self.proc(y).input_ids

    def decodes(self, y):
        with self.proc.as_target_processor():
            return self.proc.decode(y)

class ENTransformersTokenizer(Transform):
    def __init__(self, tok=None):
        self.tokenizer = tok

    def encodes(self, s: str) -> TensorText:
        toks = tensor(self.tokenizer(s)["input_ids"])
        return TensorText(toks)

    def batch_decode(self, xs, group_tokens=True):
        if len(xs.shape) == 2:
            decoded = [self.tokenizer.decode(x, group_tokens=group_tokens) for x in xs]
            no_pads = [x.replace(self.tokenizer.pad_token, "") for x in decoded]
            return no_pads
        raise Exception("xs should be a two dimensional vector if using batch_decode")

    def decodes(self, x, group_tokens=True):
        return TitledStr(
            self.tokenizer.decode(x.cpu().numpy(), group_tokens=group_tokens)
        )


class JPTransformersTokenizer(Transform):
    hira = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだじづでどばびぶべぼぱぴぷぺぽゃゅょっぁぃぅぇぉ"
    kata = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダジヅデドバビブベボパピプペポャュョッァィゥェォ"
    trans = str.maketrans(kata, hira)
    node_format_csv = r"%f[7]|"
    eos_format_csv = r"[EOS]\n"
    unk_format_csv = r"%m|"

    def __init__(self, tok=None, mcb=None):
        import MeCab

        self.tokenizer = tok
        self.mcb = mcb
        if not self.mcb:
            self.mcb = MeCab.Tagger(
                f" --node-format='{self.node_format_csv}'"
                + f" --unk-format='{self.unk_format_csv}'"
                + f" --eos-format='{self.eos_format_csv}'"
            )

    def kata2hira(self, s):
        return s.translate(self.trans).strip()

    def mecab_step(self, s: str):
        s = self.mcb.parse(s.lower())
        return "[BOS]" + self.kata2hira(s)

    def encodes(self, s: str) -> TensorText:
        s = self.mecab_step(s)
        toks = tensor(self.tokenizer(s)["input_ids"])
        return TensorText(toks)

    def batch_decode(self, xs, group_tokens=True):
        if len(xs.shape) == 2:
            decoded = [self.tokenizer.decode(x, group_tokens=group_tokens) for x in xs]
            no_pads = [x.replace(self.tokenizer.pad_token, "") for x in decoded]
            return no_pads
        raise AttributeError

    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


class Pad_Audio_Batch(ItemTransform):
    "Pad `samples` by adding padding by chunks of size `seq_len`"

    def __init__(
        self,
        pad_idx_text=1,
        pad_idx_audio=1,
        pad_first=True,
        seq_len=72,
        decode=True,
        with_attention_masks=False,
        **kwargs,
    ):
        store_attr(
            "pad_idx_text,pad_first,seq_len,seq_len,pad_idx_audio,with_attention_masks"
        )
        super().__init__(**kwargs)

    def before_call(self, b):
        "Set `self.max_len` before encodes"
        self.lens_x = [x[0].shape[1] for x in b]
        self.lens_y = [x[1].shape[0] for x in b]
        self.max_len_x = max(self.lens_x)
        self.max_len_y = max(self.lens_y)

    def __call__(self, b, **kwargs):
        self.before_call(b)
        return super().__call__(tuple(b), **kwargs)

    @staticmethod
    def pad_chunk(x, pad_idx=1, pad_first=True, seq_len=72, pad_len=10, atts=False):
        "Pad `x` by adding padding by chunks of size `seq_len`"
        zeros = torch.zeros_like
        ones = torch.ones_like
        l = pad_len - x.shape[0]
        pad_chunk = x.new_zeros((l // seq_len) * seq_len) + pad_idx
        pad_res = x.new_zeros(l % seq_len) + pad_idx
        x1 = (
            torch.cat([pad_chunk, x, pad_res])
            if pad_first
            else torch.cat([x, pad_chunk, pad_res])
        )
        if atts:
            atts = (
                torch.cat([zeros(pad_chunk), ones(x), zeros(pad_res)])
                if pad_first
                else torch.cat([ones(x), zeros(pad_chunk), zeros(pad_res)])
            )
        else:
            atts = None
        return retain_type(x1, x), atts

    def encodes(self, b):
        outs = []
        for i in range(len(b)):
            x, x_atts = self.pad_chunk(
                b[i][0][0],
                pad_idx=self.pad_idx_audio,
                pad_first=self.pad_first,
                seq_len=self.seq_len,
                pad_len=self.max_len_x,
                atts=self.with_attention_masks,
            )
            x = x[
                None, :
            ]  # This is needed because we want the input to have more directions
            y, y_atts = self.pad_chunk(
                b[i][1],
                pad_idx=self.pad_idx_text,
                pad_first=self.pad_first,
                seq_len=self.seq_len,
                pad_len=self.max_len_y,
                atts=self.with_attention_masks,
            )
            if self.with_attention_masks:
                # We put y at the end so that it is always the y in the dataloader
                outs.append(
                    (x, TensorAttention(x_atts > 0), TensorAttention(y_atts > 0), y)
                )
            else:
                outs.append((x, y))
        return outs

    def _decode(self, o):
        if isinstance(o, TensorAudio):
            return (
                [x[x != self.pad_idx_audio][None, :] for x in o] if self.decode else o
            )
        else:
            return [x[x != self.pad_idx_text] for x in o] if self.decode else o

    def decodes(self, o):
        x, y = o
        return zip(self._decode(x), self._decode(y))


class AudioBatchTransform(Transform):
    def encodes(self, r):
        t, sr = torchaudio_io.load(r["filename"])
        text = r["text"]
        return AudioPair(TensorAudio(t, sr=sr), text)

    def decodes(self, r: AudioPair):
        return AudioPair(TensorAudio(r[0], sr=r[0].sr), r[1])
