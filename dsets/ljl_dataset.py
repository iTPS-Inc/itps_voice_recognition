#!/usr/bin/env python3
from fastai.data.all import untar_data
import pandas as pd

LJ_SPEECH_URL = "https://www.dropbox.com/s/cwq264n040guqhj/LJSpeech-1.1.tar.bz2?dl=1"

def _get_ljl_data():
    p = untar_data(LJ_SPEECH_URL)
    df = pd.read_csv(p / "metadata.csv", sep="|", header=None)
    df[0] = df[0].apply(lambda x: p / "wavs" / (x + ".wav"))
    df = df.rename({0: "filename", 1: "text (numbers)", 2: "text"}, axis="columns")
    return p, df
