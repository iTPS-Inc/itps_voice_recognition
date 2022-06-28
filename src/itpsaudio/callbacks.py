#!/usr/bin/env python3
import logging as l
import neptune
import numpy as np
import torch
from fastai.callback.all import MixedPrecision, Callback
from fastai.callback.tensorboard import TensorBoardBaseCallback
from fastai.data.all import store_attr, to_float, join_path_file

import os

import wandb


class MixedPrecisionTransformers(MixedPrecision):
    def after_pred(self):
        if self.pred.dtype == torch.float16:
            self.learn.pred = to_float(self.pred)


class SeePreds(TensorBoardBaseCallback):
    order = 51

    def __init__(
        self,
        base_model_name,
        tok,
        n_iters=50,
        log_dir=None,
        neptune=False,
        n_vals=6,
        wandb=True,
    ):
        super().__init__()
        store_attr()

    def before_fit(self):
        self._setup_writer()

    def get_valid_preds(self):
        decoded_preds = []
        decoded_ys = []
        xs = []
        for i, b in enumerate(iter(self.dls.valid)):
            x, y = b[0], b[-1]
            with torch.no_grad():
                if i < self.n_vals:
                    preds = np.argmax(self.model(x).logits.detach().cpu(), axis=-1)
                    decoded_preds += self.tok.batch_decode(preds, group_tokens=True)
                    decoded_ys += self.tok.batch_decode(y, group_tokens=False)
                    xs += [x.detach().clone().cpu().numpy()]
                else:
                    break
        return (
            decoded_preds[:200],
            decoded_ys[:200],
            xs,
        )  # If the predictions are too long, neptune bugs

    @staticmethod
    def _write_wandb_audio(x, y_pred, y_true, caption):
        if not isinstance(x, list):
            x = [x_ for x_ in x]
        tab = []
        for i, (x, y, pred) in enumerate(zip(x, y_true, y_pred)):
            wandb.log(
                {
                    f"{caption} {i}: ": wandb.Audio(
                        x,
                        caption="True: {}<br>Pred: {}".format(y, pred),
                        sample_rate=16000,
                    )
                }
            )
            tab += [[y, pred]]
        wandb.log({caption: wandb.Table(columns=["y", "preds"], data=tab)})

    def after_pred(self):
        if self.iter % self.n_iters == 0:
            preds = np.argmax(self.pred.detach().cpu(), axis=-1)
            decoded_preds = self.tok.batch_decode(preds, group_tokens=True)
            decoded_ys = self.tok.batch_decode(self.yb[0], group_tokens=False)
            xs = self.xb[0].detach().clone().cpu().numpy()
            self._write_wandb_audio(xs, decoded_preds, decoded_ys, "random")
            dec_preds_valid, dec_ys_valid, xs_valid = self.get_valid_preds()
            self._write_wandb_audio(xs, dec_preds_valid, dec_ys_valid, "valid")


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


class DropPreds(Callback):
    order = 50

    def after_pred(self):
        self.learn.pred = self.pred.logits
