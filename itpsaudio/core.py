from typing import Optional

import torch
from fastai.data.all import TensorBase, fastuple, typedispatch, cast
from fastai.learner import Learner
from fastai.vision.all import get_grid, CancelStepException

from itpsaudio.utils import play_audio, show_specgram

class TransformersLearner(Learner):
    def _do_one_batch(self):
        self.pred = self.model(self.xb[0], labels=cast(self.yb[0], torch.Tensor))
        self('after_pred')
        self.loss_grad = self.pred["loss"]
        self.loss = self.loss_grad.clone()
        self.smooth_loss = self.loss_grad.clone()
        self('after_loss')
        if not self.training or not len(self.yb): return
        self('before_backward')
        self.loss_grad.backward()
        self._with_events(self.opt.step, 'step', CancelStepException)
        self.opt.zero_grad()

class TensorAudio(TensorBase):
    sr: Optional[int] = None

    def __init__(self, x, sr=None):
        self = super().__new__(TensorBase, x)
        self.sr = sr

    def show(self, ctx=None, title=None, **kwargs):
        play_audio(self, self.sr)
        if title is None:
            title = "Tensor Audio"
        return show_specgram(self.squeeze(), title=title, ctx=ctx, **kwargs)


class AudioPair(fastuple):
    def show(self, ctx=None, tok=None, **kwargs):
        audio, text = self
        if tok is not None:
            text = tok.decode(text)
        if audio.device.type == "cuda":
            audio = audio.cpu()
        if audio.ndim == 1:
            play_audio(audio[None, :], audio.sr)
            return show_specgram(audio, title=text, ctx=ctx, **kwargs)
        if audio.ndim == 2:
            play_audio(audio, audio.sr)
            return show_specgram(audio[0], title=text, ctx=ctx, **kwargs)
        play_audio(audio, audio.sr)
        return show_specgram(audio.squeeze(), title=text, ctx=ctx, **kwargs)


@typedispatch
def show_batch(
    x: AudioPair,
    y,
    samples,
    ctxs=None,
    max_n=6,
    nrows=None,
    ncols=2,
    figsize=None,
    tok=None,
    **kwargs,
):
    if figsize is None:
        figsize = (ncols * 6, max_n // ncols * 3)
    if ctxs is None:
        ctxs = get_grid(
            min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize
        )
    for i, ctx in enumerate(ctxs):
        print(AudioPair)
        AudioPair(x[0][i], x[-1][i]).show(ctx=ctx, tok=tok)
