from functools import lru_cache

import torch
import torchaudio
import torchaudio.backend.sox_io_backend as torchaudio_io
import torchaudio.transforms as T
from fastai.data.all import ItemTransform, Transform, noop, retain_type

from itpsaudio.core import AudioPair, TensorAttention, TensorAudio
from typing import Union


@Transform
def extract_first(s: TensorAudio):
    if s.shape[0] == 1:
        return s[0]
    else:
        return s


class Resampler(Transform):
    samplers = {16000: noop}

    def __init__(self, unique_srs):
        for sr in unique_srs:
            self.samplers[sr] = T.Resample(sr, 16000)

    def encodes(self, x: TensorAudio, sr: Union[int,None] = None):
        sr = x.sr if x.sr else sr if sr else None
        assert sr, "Please give us some information about the sampling rate"
        return TensorAudio(self.samplers[sr](x), sr=16000)


@Transform
def capitalize(s: str):
    return s.upper()


@lru_cache(maxsize=None)
def get_audio_length(s):
    t, sr = torchaudio_io.load(s)
    return len(t[0]) / sr, sr


@Transform
def squeeze(s):
    return s.squeeze()


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
        self.pad_idx_text = pad_idx_text
        self.pad_idx_audio = pad_idx_audio
        self.pad_first = pad_first
        self.seq_len = seq_len
        self.decode = decode
        self.with_attention_masks = with_attention_masks
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
                assert x_atts is not None and y_atts is not None
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
    def __init__(self, do_normalize=True):
        self.do_norm = do_normalize

    def encodes(self, r):
        t, sr = torchaudio_io.load(r["filename"], normalize=self.do_norm)
        text = r["text"]
        return AudioPair(TensorAudio(t, sr=sr), text)

    def decodes(self, r: AudioPair):
        return AudioPair(TensorAudio(r[0], sr=r[0].sr), r[1])
