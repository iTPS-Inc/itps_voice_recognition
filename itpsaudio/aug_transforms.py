#!/usr/bin/env python3
import torchaudio
from fastai.data.all import Transform

from itpsaudio.core import TensorAudio


# TODO: Give this transform an appropriate name
class AddNoise(Transform):
    def encodes(self, x: TensorAudio):
        # Define effects
        self.effects = [
            ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
            ["speed", "0.8"],  # reduce the speed
            ["rate", f"{x.sr}"],
            ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        ]
        t, sr = torchaudio.sox_effects.apply_effects_tensor(x, x.sr, self.effects)
        return TensorAudio(t, sr=sr)
