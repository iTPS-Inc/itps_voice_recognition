#!/usr/bin/env python3
from itpsaudio.core import AudioTensor
import torch

def test_audiotensor():
    assert AudioTensor([]) is not None
    assert AudioTensor([], sr=16000) is not None
