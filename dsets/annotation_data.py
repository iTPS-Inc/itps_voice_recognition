#!/usr/bin/env python3
from ctypes import ArgumentError
from fastai.data.all import untar_data
import pandas as pd

ANNOTATION_DATA_URL = "https://www.dropbox.com/s/92tbhcdlymk5s0w/annotation_data.tar.gz?dl=1"

def get_annotation_data(lang=None):
    p = untar_data(ANNOTATION_DATA_URL)
    p = p / "アノテーションデータ"/ "20210511_アノテーションデータ_full"
    df = pd.read_csv(p / "annotation_data.csv", index_col=0)
    print(df)
    df["filename"] = df["audio_filename"].apply(lambda x: p / x)
    df["lang"] = df["audio_filename"].apply(lambda x: x.split("/")[1])
    if lang is None:
        return df
    elif lang =="en":
        return p, df[df["lang"] == lang].reset_index(drop=True)
    elif lang == "jp":
        return p, df[df["lang"] == lang].reset_index(drop=True)
    else:
        raise ArgumentError
