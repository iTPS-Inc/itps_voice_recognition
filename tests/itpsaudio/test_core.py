#!/usr/bin/env python3
from itpsaudio.core import AudioTensor
from fastai.vision.all import TensorBase
import torch


def test_can_create_empty_audiotensor():
    empty_tensor = AudioTensor([])
    assert empty_tensor is not None
    assert empty_tensor.sr is None
    assert isinstance(empty_tensor, AudioTensor)
    assert isinstance(empty_tensor, TensorBase)

    tensor_with_sr = AudioTensor([], sr=16000)
    assert tensor_with_sr is not None
    assert isinstance(tensor_with_sr, AudioTensor)


def test_AudioTensor_operates_like_tensor():
    tensor_ones = AudioTensor([1, 1], sr=16000)
    tensor_zeros = torch.zeros_like(tensor_ones)
    assert (tensor_ones - tensor_ones == tensor_zeros).all()
