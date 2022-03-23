#!/usr/bin/env python3
from itpsaudio.core import AudioTensor
from itpsaudio.transforms import Pad_Audio_Batch, capitalize
from fastai.vision.all import tensor
import torch


def _assert_equal_tensor(a, b):
    assert (a == b).all()

def test_padding_Audio_Batch_works():
    xy_1 = (AudioTensor([[1, 1, 1]], sr=16000), tensor([1, 2, 3, 4, 5, 6]))
    xy_2 = (AudioTensor([[1, 1]], sr=16000), tensor([1, 2, 3]))
    b = (xy_1, xy_2)
    pad = Pad_Audio_Batch(
        pad_idx_audio=-100, pad_idx_text=-100, pad_first=True, seq_len=1, decode=True
    )
    ((p_x1,p_y1),(p_x2, p_y2)) = pad(b)
    _assert_equal_tensor(p_x1, AudioTensor([[1, 1, 1]], sr=16000))
    _assert_equal_tensor(p_x2, AudioTensor([[-100, 1, 1]], sr=16000))
    _assert_equal_tensor(p_y1, tensor([[1   ,    2,   3, 4, 5, 6]]))
    _assert_equal_tensor(p_y2, tensor([[-100, -100,-100, 1, 2, 3]]))

def test_pad_first():
    xy_1 = (AudioTensor([[1, 1, 1]], sr=16000), tensor([1, 2, 3, 4, 5, 6]))
    xy_2 = (AudioTensor([[1, 1]], sr=16000), tensor([1, 2, 3]))
    b = (xy_1, xy_2)
    pad = Pad_Audio_Batch(
        pad_idx_audio=-100, pad_idx_text=-100, decode=True, pad_first=True
    )
    ((p_x1,p_y1),(p_x2, p_y2)) = pad(b)
    _assert_equal_tensor(p_x1, AudioTensor([[1, 1, 1]], sr=16000))
    _assert_equal_tensor(p_x2, AudioTensor([[1, 1, -100]], sr=16000))
    _assert_equal_tensor(p_y1, tensor([[1   ,    2,   3, 4, 5,    6]]))
    _assert_equal_tensor(p_y2, tensor([[1, 2, 3, -100, -100, -100]]))
