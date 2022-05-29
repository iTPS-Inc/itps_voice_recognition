#!/usr/bin/env python3
from fastai.callback.all import MixedPrecision
from fastai.data.all import to_float, Callback, Recorder, store_attr
from fastai.callback.tensorboard import TensorBoardBaseCallback
import torch
import neptune
from  neptune.new.types import File
import numpy as np


class MixedPrecisionTransformers(MixedPrecision):
    def after_pred(self):
        if self.pred.logits.dtype == torch.float16:
            self.learn.pred.logits = to_float(self.pred.logits)

class SeePreds(TensorBoardBaseCallback):
    def __init__(self,base_model_name,tok, n_iters=50, log_dir=None, neptune=False, n_vals=6):
      super().__init__()
      store_attr()

    def before_fit(self):
       self._setup_writer()
       if self.neptune:
         self.experiment = neptune.get_experiment()

    def get_valid_preds(self):
      decoded_preds = []
      decoded_ys = []
      for i, (x, y) in enumerate(iter(self.dls.valid)):
        if i < self.n_vals:
          with torch.no_grad():
             preds = np.argmax(self.model(x).logits.detach().cpu(), axis=-1)
          decoded_preds += self.tok.batch_decode(preds)
          decoded_ys += self.tok.batch_decode(y)
        else:
          break
      return decoded_preds, decoded_ys

    def _log_preds_to_path(self, p, y, path):
      self.writer.add_text(f"{self.base_model_name}/prediction/{path}",f"Targets:{y}\nPredictions:{p}",  self.iter)
      if self.neptune:
         self.experiment.log_text(f"text_predictions/{path}", f"{self.iter}: Targ: {y}")
         self.experiment.log_text(f"text_predictions/{path}", f"{self.iter}: Pred: {p}")

    def after_pred(self):
        if self.iter % self.n_iters == 0:
            preds = np.argmax(self.pred.logits.detach().cpu(), axis=-1)
            decoded_preds = self.tok.batch_decode(preds)
            decoded_ys = self.tok.batch_decode(self.yb[0])
            dec_preds_valid,  dec_ys_valid = self.get_valid_preds()
            for x in self.xb[0]:
              self.writer.add_audio(f"{self.iter}: Audio", x, self.iter, 16000)

            for p,y in zip(decoded_preds, decoded_ys):
              self._log_preds_to_path(p,y, "random")

            for i, (p,y) in enumerate(zip(dec_preds_valid, dec_ys_valid)):
              self._log_preds_to_path(p,y, f"valid_{i}")
