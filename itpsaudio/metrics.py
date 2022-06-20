#!/usr/bin/env python3
import numpy as np
import jiwer

class WER():
    def __init__(self, tok): self.tok = tok

    def __call__(self, pred, labels):
        pred_logits = pred.logits
        pred_ids = np.argmax(pred_logits.detach().cpu().numpy(), axis=-1)
        pred_str = self.tok.batch_decode(pred_ids, group_tokens=True)
        label_str = self.tok.batch_decode(labels, group_tokens=False)
        wer = jiwer.wer(label_str, pred_str)
        return wer

class CER():
    def __init__(self, tok): self.tok = tok

    def __call__(self, pred, labels):
        pred_logits = pred.logits
        pred_ids = np.argmax(pred_logits.detach().cpu().numpy(), axis=-1)
        pred_str = self.tok.batch_decode(pred_ids, group_tokens=True)
        label_str = self.tok.batch_decode(labels, group_tokens=False)
        cer = jiwer.cer(label_str, pred_str)
        return cer
