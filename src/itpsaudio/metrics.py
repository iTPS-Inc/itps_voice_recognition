#!/usr/bin/env python3
import numpy as np
import jiwer
from fastai.metrics import Metric

class WER(Metric):
  def __init__(self, tok):
      self.metric_name = "wer"
      self.tok = tok
      self.pred_strs, self.label_strs = [], []

  @property
  def name(self):
      return "wer"

  def accumulate(self, learn):
      pred_logits = learn.pred
      labels = learn.yb[-1].detach().cpu().numpy()
      pred_ids = np.argmax(pred_logits.detach().cpu().numpy(), axis=-1)
      pred_str = self.tok.batch_decode(
          pred_ids, group_tokens=True, skip_special_tokens=False
      )
      label_str = self.tok.batch_decode(
          labels, group_tokens=False, skip_special_tokens=False
      )
      pred_str = [i if i != "" else "[NONE]" for i in pred_str]
      self.pred_strs += pred_str
      self.label_strs += label_str

  def reset(self):
      self.pred_strs, self.label_strs = [], []

  @property
  def value(self):
      cer = jiwer.wer(self.pred_strs, self.label_strs)
      return cer

class CER(Metric):
  def __init__(self, tok):
      self.metric_name = "cer"
      self.tok = tok
      self.pred_strs, self.label_strs = [], []

  @property
  def name(self):
      return "cer"

  def accumulate(self, learn):
      pred_logits = learn.pred
      labels = learn.yb[-1].detach().cpu().numpy()
      pred_ids = np.argmax(pred_logits.detach().cpu().numpy(), axis=-1)
      pred_str = self.tok.batch_decode(
          pred_ids, group_tokens=False, skip_special_tokens=False
      )
      label_str = self.tok.batch_decode(
          labels, group_tokens=False, skip_special_tokens=False
      )
      pred_str = [i if i != "" else "[NONE]" for i in pred_str]

      self.pred_strs += pred_str
      self.label_strs += label_str


  @property
  def value(self):
      cer = jiwer.cer(self.pred_strs, self.label_strs)
      return cer

  def reset(self):
      self.pred_strs, self.label_strs = [], []
