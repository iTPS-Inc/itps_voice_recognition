from typing import Optional

import torch
from fastai.data.all import TensorBase, fastuple, typedispatch, cast
from fastai.learner import Learner
from fastai.vision.all import get_grid, CancelStepException

from itpsaudio.utils import play_audio, show_specgram

class TransformersLearnerAtt(Learner):
    def _do_one_batch(self):
        self.pred = self.model(self.xb[0],
                               attention_mask=self.xb[1],
                               output_attentions=False,
                               labels=cast(self.yb[-1], torch.Tensor))
        self.loss_grad = self.pred["loss"]
        self("after_pred")
        self.loss = self.loss_grad.clone()
        self.smooth_loss = self.loss_grad.clone()
        self("after_loss")
        if not self.training or not len(self.yb):
            return
        self("before_backward")
        self.loss_grad.backward()
        self._with_events(self.opt.step, "step", CancelStepException)
        self.opt.zero_grad()

class TransformersLearnerOwnLoss(Learner):
    def __init__(self, *args,with_attentions=True, **kwargs):
      super().__init__(*args, **kwargs)
      if with_attentions:
        self._do_one_batch = self._do_one_batch_w_att
      else:
        self._do_one_batch = self._do_one_batch_wo_att

    def _do_one_batch_wo_att(self):
        self.pred = self.model(self.xb[0])
        self("after_pred")
        if len(self.yb):
            self.loss_grad = self.loss_func(
                preds=cast(self.pred, torch.Tensor),
                inp_len=torch.ones_like(self.xb[0], dtype=torch.long).sum(-1),
                labels=cast(self.yb[-1], torch.Tensor),
                modelconf=self.model.config,
            )
            self.loss = self.loss_grad.clone()
        self("after_loss")
        if not self.training or not len(self.yb):
            return
        self("before_backward")
        self.loss_grad.backward()
        self._with_events(self.opt.step, "step", CancelStepException)
        self.opt.zero_grad()

    def _do_one_batch_w_att(self):
        # TODO: When using spec mask, can't use attention_mask
        self.pred = self.model(self.xb[0], attention_mask=self.xb[1])
        self("after_pred")
        if len(self.yb):
            self.loss_grad = self.loss_func(
                preds=cast(self.pred, torch.Tensor),
                inp_len=self.xb[1].sum(1).long(),
                labels=cast(self.yb[0], torch.Tensor),
                modelconf=self.model.config,
            )
            self.loss = self.loss_grad.clone()
        self("after_loss")
        if not self.training or not len(self.yb):
            return
        self("before_backward")
        self.loss_grad.backward()
        self._with_events(self.opt.step, "step", CancelStepException)



class TransformersLearner(Learner):
    def _do_one_batch(self):
        self.pred = self.model(self.xb[0], labels=cast(self.yb[-1], torch.Tensor))
        self.loss_grad = self.pred["loss"]
        self("after_pred")
        self.loss = self.loss_grad.clone()
        self.smooth_loss = self.loss_grad.clone()
        self("after_loss")
        if not self.training or not len(self.yb):
            return
        self("before_backward")
        self.loss_grad.backward()
        self._with_events(self.opt.step, "step", CancelStepException)
        self.opt.zero_grad()


class TensorAttention(TensorBase):
    def __init__(self, x):
        self = super().__new__(TensorBase, x)


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
    def show(self, ctx=None, tok=None,skip_special_tokens=False, **kwargs):
        audio, text = self
        if tok is not None:
            text = tok.decode(text, group_tokens=False,
                              skip_special_tokens=skip_special_tokens)
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
