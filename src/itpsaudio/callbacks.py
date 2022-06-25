#!/usr/bin/env python3
import logging as l
import neptune
import numpy as np
import torch
from fastai.callback.all import MixedPrecision, Callback
from fastai.callback.tensorboard import TensorBoardBaseCallback
from fastai.data.all import store_attr, to_float, join_path_file

import os


class MixedPrecisionTransformers(MixedPrecision):
    def after_pred(self):
        if self.pred.dtype == torch.float16:
            self.learn.pred = to_float(self.pred)


class SeePreds(TensorBoardBaseCallback):
    def __init__(
        self, base_model_name, tok, n_iters=50, log_dir=None, neptune=False, n_vals=6
    ):
        super().__init__()
        store_attr()

    def before_fit(self):
        self._setup_writer()
        if self.neptune:
            self.experiment = neptune.get_experiment()

    def get_valid_preds(self):
        decoded_preds = []
        decoded_ys = []
        for i, b in enumerate(iter(self.dls.valid)):
            if len(b) == 4:
                x, y = b[0], b[-1]
            elif len(b) == 2:
                x, y = b[0], b[-1]
            else:
                assert False, "Don't know how to See these preds"

            if i < self.n_vals:
                with torch.no_grad():
                    preds = np.argmax(self.model(x).detach().cpu(), axis=-1)
                decoded_preds += self.tok.batch_decode(preds, group_tokens=True)
                decoded_ys += self.tok.batch_decode(y, group_tokens=False)
            else:
                break
        return (
            decoded_preds[:200],
            decoded_ys[:200],
        )  # If the predictions are too long, neptune bugs

    def _log_preds_to_path(self, p, y, path):
        self.writer.add_text(
            f"{self.base_model_name}/prediction/{path}",
            f"Targets:{y}\nPredictions:{p}",
            self.iter,
        )
        if self.neptune:
            self.experiment.log_text(
                f"text_predictions/{path}", f"{self.iter}: Targ: {y}"
            )
            self.experiment.log_text(
                f"text_predictions/{path}", f"{self.iter}: Pred: {p}"
            )

    def after_pred(self):
        if self.iter % self.n_iters == 0:
            preds = np.argmax(self.pred.detach().cpu(), axis=-1)
            decoded_preds = self.tok.batch_decode(preds, group_tokens=True)
            decoded_ys = self.tok.batch_decode(self.yb[0], group_tokens=False)
            dec_preds_valid, dec_ys_valid = self.get_valid_preds()
            for x in self.xb[0]:
                self.writer.add_audio(f"{self.iter}: Audio", x, self.iter, 16000)

            for p, y in zip(decoded_preds, decoded_ys):
                self._log_preds_to_path(p, y, "random")

            for i, (p, y) in enumerate(zip(dec_preds_valid, dec_ys_valid)):
                self._log_preds_to_path(p, y, f"valid_{i}")


class NeptuneSaveModel(Callback):
    def __init__(self, n_epochs):
        super().__init__()
        store_attr()

    def before_fit(self):
        self.experiment = neptune.get_experiment()

    def after_epoch(self):
        if self.iter % self.n_epochs == 0:
            _file = join_path_file(
                self.learn.save_model.fname,
                self.learn.path / self.learn.model_dir,
                ext=".pth",
            )
            if os.path.exists(_file):
                neptune.log_artifact(str(_file))
            else:
                l.log(
                    l.WARNING,
                    f"Could find {_file}, thus couldn't upload it to Neptune.",
                )
