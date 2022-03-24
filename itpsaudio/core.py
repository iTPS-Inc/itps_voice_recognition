from typing import Optional

from fastai.data.all import TensorBase, fastuple

from itpsaudio.utils import play_audio, show_specgram


class TensorAudio(TensorBase):
    sr: Optional[int] = None

    def __init__(self, x, sr=None):
        self = super().__new__(TensorBase, x)
        self.sr = sr


class AudioPair(fastuple):
    def show(self, ctx=None, tok=None, **kwargs):
        audio, text = self
        if tok is not None:
            text = tok.decode(text)
        if audio.device.type == "cuda":
            audio = audio.cpu()
        if audio.ndim == 2:
            play_audio(audio[None, :], audio.sr)
            return show_specgram(audio, title=text, ctx=ctx, **kwargs)
        elif audio.ndim == 3:
            play_audio(audio, audio.sr)
            return show_specgram(audio.squeeze(), title=text, ctx=ctx, **kwargs)
