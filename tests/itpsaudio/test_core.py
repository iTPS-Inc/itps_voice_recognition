#!/usr/bin/env python3
import os

import torch
import torchaudio
from fastai.vision.all import Path, TensorBase

from itpsaudio.core import TensorAudio

_SAMPLE_DIR = "_assets"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")
TEST_DIR = Path(os.path.abspath(__file__)).parent

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
    empty_tensor = TensorAudio([])
    assert empty_tensor is not None
    assert empty_tensor.sr is None
    assert isinstance(empty_tensor, TensorAudio)
    assert isinstance(empty_tensor, TensorBase)

    tensor_with_sr = TensorAudio([], sr=16000)
    assert tensor_with_sr is not None
    assert isinstance(tensor_with_sr, TensorAudio)


def test_AudioTensor_operates_like_tensor():
    tensor_ones = TensorAudio([1, 1], sr=16000)
    tensor_zeros = torch.zeros_like(tensor_ones)
    assert (tensor_ones - tensor_ones == tensor_zeros).all()

