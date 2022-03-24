#!/usr/bin/env python3
import torch
from fastai.vision.all import tensor, TfmdLists

from itpsaudio.core import TensorAudio
from itpsaudio.utils import splits_testcol
from itpsaudio.transforms import Pad_Audio_Batch, AudioBatchTransform
from dsets.dset_collection.simple_dsets import get_test_data


def _assert_equal_tensor(a, b):
    assert (a == b).all()

def test_Pad_Audio_Batch():
    xy_1 = (TensorAudio([[1, 1, 1]], sr=16000), tensor([1, 2, 3, 4, 5, 6]))
    xy_2 = (TensorAudio([[1, 1]], sr=16000), tensor([1, 2, 3]))
    b = (xy_1, xy_2)
    pad = Pad_Audio_Batch(
        pad_idx_audio=-100, pad_idx_text=-100, pad_first=True, seq_len=1, decode=True
    )
    ((p_x1,p_y1),(p_x2, p_y2)) = pad(b)
    _assert_equal_tensor(p_x1, TensorAudio([[1, 1, 1]], sr=16000))
    _assert_equal_tensor(p_x2, TensorAudio([[-100, 1, 1]], sr=16000))
    _assert_equal_tensor(p_y1, tensor([[1   ,    2,   3, 4, 5, 6]]))
    _assert_equal_tensor(p_y2, tensor([[-100, -100,-100, 1, 2, 3]]))

def test_Pad_Audio_Batch_pad_first():
    xy_1 = (TensorAudio([[1, 1, 1]], sr=16000), tensor([1, 2, 3, 4, 5, 6]))
    xy_2 = (TensorAudio([[1, 1]], sr=16000), tensor([1, 2, 3]))
    b = (xy_1, xy_2)
    pad = Pad_Audio_Batch(
        pad_idx_audio=-100, pad_idx_text=-100, decode=True, pad_first=True
    )
    ((p_x1,p_y1),(p_x2, p_y2)) = pad(b)
    _assert_equal_tensor(p_x1, TensorAudio([[1, 1, 1]], sr=16000))
    _assert_equal_tensor(p_x2, TensorAudio([[1, 1, -100]], sr=16000))
    _assert_equal_tensor(p_y1, tensor([[1   ,    2,   3, 4, 5,    6]]))
    _assert_equal_tensor(p_y2, tensor([[1, 2, 3, -100, -100, -100]]))


def test_Pad_Audio_Batch_decode():
    xy_1 = (TensorAudio([[1, 1, 1]], sr=16000), tensor([1, 2, 3, 4, 5, 6]))
    xy_2 = (TensorAudio([[1, 1]], sr=16000), tensor([1, 2, 3]))
    b = (xy_1, xy_2)
    pad = Pad_Audio_Batch(
        pad_idx_audio=-100, pad_idx_text=-100, decode=True, pad_first=True
    )
    padded_b = pad(b)
    xs,ys = [], []
    for x,y in padded_b: xs.append(x); ys.append(y[None,:])
    xs = torch.concat(xs)
    ys = torch.concat(ys)
    padded_b= (xs, ys)
    dec_b = pad.decode(padded_b)
    print(padded_b)
    for (dx,dy),(x, y) in zip(dec_b, b):
        _assert_equal_tensor(dx, x)
        _assert_equal_tensor(dy, y)

def test_AudioBatchTransform():
    p, df = get_test_data()
    abt = AudioBatchTransform(p, df)
    splits = splits_testcol(df)
    tfms = TfmdLists(df, abt, splits=splits)
    dls = tfms.dataloaders(bs=1)
