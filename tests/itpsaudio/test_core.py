#!/usr/bin/env python3
import os

import torch
import torchaudio
from fastai.vision.all import Path, TensorBase

import IPython.display
from itpsaudio.core import AudioTensor, LoadAudio

_SAMPLE_DIR = "_assets"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")

def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_PATH, resample=resample)


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


def test_LoadAudio():
    load = LoadAudio(path=Path("_assets"))
    x = load("speech.wav")
    assert isinstance(x, AudioTensor)
    assert x.sr == 16000
