#!/usr/bin/env python3
from fastai.callback.all import MixedPrecision
from fastai.data.all import to_float
import torch


class MixedPrecisionTransformers(MixedPrecision):
    def after_pred(self):
        if self.pred.logits.dtype == torch.float16:
            self.learn.pred.logits = to_float(self.pred.logits)
