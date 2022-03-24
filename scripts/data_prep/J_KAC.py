#!/usr/bin/env python3
import os, shutil
from pathlib import Path

import pandas as pd
from fastai.data.all import get_files, untar_data
from fastai.data.transforms import RandomSplitter
from tqdm import tqdm

from dsets.helpers.helpers import make_tarfile, train_test_split

J_KAC_URL_ORIG = "https://www.dropbox.com/s/x1nyr7hd23d5128/J-KAC.tar.gz?dl=1"
J_KAC_TXTS = "https://www.dropbox.com/s/id68k7dq1fu78gq/J-KAC_txt.tar.gz?dl=1"

OUTPATH = (
    "/home/jjs/Dropbox/Share to iTPS AI-Team/train_data_repository/J_KAC.tar.gz"
)


def _make_destination(p, fn, test):
    if test:
        split = "test"
    else:
        split = "train"
    if str(fn).startswith("ru/"):
        return p / fn.parent.parent.parent / split / fn.name
    return p / fn.parent.parent / split / fn.name

def get_lang_frame(info_file, label_file):
    i_df = pd.read_csv(info_file, sep="\t")
    if not isinstance(i_df, pd.DataFrame):
        raise TypeError(
            "Could not read pandas dataframe of label file in nict spreds dataset."
        )

    l_df = pd.read_csv(label_file, sep="\t", header=None)
    if not isinstance(l_df, pd.DataFrame):
        raise TypeError(
            "Could not read pandas dataframe of label file in nict spreds dataset."
        )
    l_df = l_df.rename({0: "filename", 1: "text"}, axis="columns")
    l_df = l_df.set_index("filename")["text"].to_dict()
    i_df["text"] = i_df["wav"].apply(lambda x: l_df[x])
    i_df["filename"] = i_df["language"].apply(lambda x: x + "/") + i_df["wav"]
    i_df = train_test_split(i_df)
    return i_df



def move_files_into_splits(df, p):
    for i in p.ls():
        if not os.path.isdir(i):
            continue
        if not os.path.exists(i / "WAVE" / "train"):
            os.mkdir(i / "WAVE" / "train")
        if not os.path.exists(i / "WAVE" / "test"):
            os.mkdir(i / "WAVE" / "test")
    for i, r in df.iterrows():
        if r["test"]:
            src = p / r["filename"]
            dest = _make_destination(p, r["filename"], True)
            if os.path.exists(src) and not os.path.exists(dest):
                shutil.copy(src, dest)
        elif not r["test"]:
            src = p / r["filename"]
            dest = _make_destination(p, r["filename"], False)
            if os.path.exists(src) and not os.path.exists(dest):
                shutil.copy(src, dest)
    lang = df["language"].iloc[0]
    if df["filename"].apply(os.path.exists).all():
        for d in (p / lang / "WAVE").ls():
            if not d.name in ["train", "test"]:
                print(d)
                shutil.rmtree( d )
    df["filename"] = df[["filename", "test"]].apply(
        lambda x: _make_destination(Path("."), x[0], x[1]), axis=1
    )
    return df

def get_jkac_data(force_download=False):
    p = untar_data(J_KAC_URL_ORIG, force_download=True)
    p_txt = untar_data(J_KAC_TXTS, force_download=True)


    if os.path.exists(p / "doc"):
        shutil.rmtree(p / "doc") label_files = get_files(p, extensions=[".label"])
    info_files = get_files(p, extensions=[".info"])
    dfs = []
    for label_file, info_file in tqdm(zip(label_files, info_files)):
        df = get_lang_frame(info_file, label_file)
        df = move_files_into_splits(df, p)
        dfs.append(df)
    out_df = pd.concat(dfs, ignore_index=True)
    out_df.loc[out_df["language"] == "ja", "language"] = "jp"
    return p, out_df

p, df = get_nict_data()
df.to_csv(p / "metadata.csv")
make_tarfile(OUTPATH, p)
