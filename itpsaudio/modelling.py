from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
from typing import Union, Optional


def torch_int_div(a, b):
    return torch.div(a, b, rounding_mode="floor")


def get_feat_extract_output_lengths(
    modelconf,
    input_lengths: Union[torch.LongTensor, int],
    add_adapter: Optional[bool] = None,
):
    """
    Computes the output length of the convolutional layers
    """
    add_adapter = modelconf.add_adapter if add_adapter is None else add_adapter

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return torch_int_div(input_length - kernel_size, stride) + 1

    for kernel_size, stride in zip(modelconf.conv_kernel, modelconf.conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    if add_adapter:
        for _ in range(modelconf.num_adapter_layers):
            input_lengths = _conv_out_length(input_lengths, 1, modelconf.adapter_stride)

    return input_lengths


class CTCLoss(_Loss):
    def __init__(self, num_classes, blank=0, weight=0.01, reduction="mean"):
        """
        Small CTCLoss class.
        calculating the ctcloss similarly to how transformers does it "
        """
        super().__init__(reduction=reduction)
        self.weight = weight
        self.num_classes = num_classes
        self.ctc = nn.CTCLoss(reduction=reduction, blank=blank, zero_infinity=True)

    def forward(self, preds, inp_len, labels, modelconf):
        inp_len = get_feat_extract_output_lengths(modelconf, inp_len, False)
        labels_mask = labels > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)
        log_probs = nn.functional.log_softmax(
            preds, dim=-1, dtype=torch.float32
        ).transpose(0, 1)
        ctc_loss = self.ctc(log_probs, flattened_targets, inp_len, target_lengths)
        return ctc_loss


class SmoothCTCLoss(_Loss):
    def __init__(self, num_classes, blank=0, weight=0.01, ctc_red="mean", kl_red="mean"):
        """
        CTC loss with label smoothing.
        """
        super().__init__()
        self.weight = weight
        if weight is None:
            self.weight=0.01
        self.num_classes = num_classes
        self.ctc = nn.CTCLoss(reduction=ctc_red, blank=blank, zero_infinity=True)
        kldiv_red = "batchmean" if kl_red=="mean" else "sum"
        self.kldiv = nn.KLDivLoss(reduction=kldiv_red)

    def forward(self, preds, inp_len, labels, modelconf):
        inp_len = get_feat_extract_output_lengths(modelconf, inp_len, False)
        labels_mask = labels > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)
        log_probs = nn.functional.log_softmax(
            preds, dim=-1, dtype=torch.float32
        ).transpose(0, 1)
        ctc_loss = self.ctc(log_probs, flattened_targets, inp_len, target_lengths)
        kl_inp = log_probs.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 1.0 / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)
        loss = (1.0 - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss
