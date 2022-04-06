import math
import random

import torch
import torchaudio
from fastai.vision.augment import RandTransform

from itpsaudio.core import TensorAudio


class RandomReverbration(RandTransform):
    effects = [["reverb", "-w"]]
    def encodes(self, x: TensorAudio):
        assert (
            x.sr is not None
        ), "Tensor passed to RandomReverbration should have sample rate"
        t, sr = torchaudio.sox_effects.apply_effects_tensor(x, x.sr, self.effects)
        return TensorAudio(t, sr=sr)


class AddNoise(RandTransform):
    def __init__(self, db_range, noise, power=2, **kwargs):
        super().__init__(**kwargs)
        self.db_range = db_range
        self.noise = noise
        self.power = power
        self.noise_power = torch.norm(noise, p=power)

    def encodes(self, speech: TensorAudio, for_show=False):
        # If noise is shorted than input audio, concat it onto itself to make it longer:
        while self.noise.shape[-1] < speech.shape[-1]:
            self.noise = torch.concat([self.noise, self.noise], dim=-1)
        speech_length = speech.shape[-1]
        noise_start = self.noise.shape[-1] - speech_length
        noise_sample_start = random.randrange(0, noise_start)

        # Randomly determine loudness of the noise
        db = random.uniform(self.db_range.start, self.db_range.stop)
        snr = math.exp(db / 10)
        if for_show:
            print("Signal Noise Ratio: ", snr)
        speech_power = torch.norm(speech, p=self.power)
        scale = snr * self.noise_power / speech_power
        scale = (
            scale * speech
            + self.noise[:, noise_sample_start : noise_sample_start + speech_length]
        ) / 2
        return TensorAudio(scale, sr=speech.sr)
