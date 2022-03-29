#!/usr/bin/env python3
import torch
from fastai.vision.all import tensor, TfmdLists

from itpsaudio.core import TensorAudio
from itpsaudio.aug_transforms import AddNoise
from dsets.dset_collection.simple_dsets import get_test_data
import requests
import torchaudio

SAMPLE_NOISE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"

with open("noise_sample.wav", "wb") as f:
    r = requests.get(SAMPLE_NOISE_URL)
    f.write(r.content)

def _assert_equal_tensor(a, b):
    assert (a == b).all()

def _assert_nequal_tensor(a, b):
    assert not (a == b).all()

def test_AudioBatchTransform():
    xy_1 = TensorAudio([[1, 1, 1]], sr=16000).float()
    noise, sr = torchaudio.load("noise_sample.wav")
    noise = TensorAudio(noise, sr=sr)
    noise_t = AddNoise(range(1, 10), noise, p=1)

    x = noise_t(xy_1)
    _assert_nequal_tensor(x, xy_1)
