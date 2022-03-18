#!/usr/bin/env python3
import os
import shutil
import subprocess

import pandas as pd
from fastai.data.all import get_files, untar_data
from tqdm import tqdm

from helpers import make_tarfile


def is_this_ok():
    answer = input("Is this OK with you? [y/n]")
    if answer == "y":
        return True
    else:
        return False


def convert_mp3_wav(inn, outn):
    res = subprocess.run(
        [
            "ffmpeg",
            "-i",
            inn,
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            f"{outn}",
            "-y",
        ]
    )
    return res


ANNOTATION_DATA_URL_ORIG = (
    "https://www.dropbox.com/s/qzvrx0c3rrmxxl3/annotation_data_initial.tar.gz?dl=1"
)
OUT_PATH = "/home/jjs/Dropbox/Share to iTPS AI-Team/train_data_repository/annotation_data.tar.gz"


def main(force_download=False):
    p = untar_data(ANNOTATION_DATA_URL_ORIG, force_download=force_download)
    print("Finished download")
    jp_path = p / "20210511_Japanese_Delivery_Fin"
    if os.path.exists(jp_path):
        shutil.move(str(jp_path), p / "jp/")

    en_path = p / "20210511_English_Delivery_Fin"
    if os.path.exists(en_path):
        shutil.move(str(en_path), p / "en/")
    print("Prepared directories")
    dfs = []
    for lang in ["en", "jp"]:
        wav_files = get_files(p / lang, extensions=[".wav"])
        csv_files = get_files(p / lang, extensions=[".csv"])
        df = pd.concat([pd.read_csv(i) for i in csv_files]).reset_index()
        df[["st", "text", "et"]] = df["Transcription"].str.extract(
            r"(\[.*\])(.*)(\[.*\])"
        )
        a = {}
        for i in wav_files:
            a[i.name] = i
        df["filename"] = df["Segmented audio file name"].apply(lambda x: a[x])
        df["lang"] = lang
        train = df.sample(frac=0.8, random_state=123).index
        df.loc[:, "train"] = False
        df.loc[train, "train"] = True
        df["test"] = ~df["train"]
        for ind, r in tqdm(df.iterrows()):
            if r["test"]:
                src = r["filename"]
                langfolder = src.parent.parent
                dest = langfolder / "test" / src.name
                if not os.path.exists(dest.parent):
                    os.mkdir(dest.parent)
                if os.path.exists(dest):
                    print("continuing")
                    continue
                convert_mp3_wav(src, dest)

            if not r["test"]:
                src = r["filename"]
                langfolder = src.parent.parent
                dest = langfolder / "train" / src.name
                if not os.path.exists(dest.parent):
                    os.mkdir(dest.parent)
                if os.path.exists(dest):
                    print("continuing")
                    continue
                convert_mp3_wav(src, dest)

        dfs = dfs + [df]
        print(f"Finished moving {lang} files")

    df = pd.concat(dfs).drop("index", axis="columns").reset_index(drop=True)
    df["dataset"] = df["train"].apply(lambda x: "train" if x else "test")
    df["filename"] = df.apply(
        lambda x: x["filename"].parent.parent / x["dataset"] / x["filename"].name,
        axis=1,
    )

    ok = df["filename"].apply(os.path.exists).all()
    if not ok:
        print(
            f"{(~df['filename'].apply(os.path.exists)).sum()} files could not be converted correctly"
        )
        ok = is_this_ok()
        if ok:
            working_files = df["filename"].apply(os.path.exists)
            df = df[working_files].copy()
    if ok:
        for d in (p / "en").ls():
            if not d.name in ["train", "test"]:
                print(d)
                shutil.rmtree(d)
        for d in (p / "jp").ls():
            if not d.name in ["train", "test"]:
                print(d)
                shutil.rmtree(d)
        df.to_csv(p / "annotation_data.csv")
        make_tarfile(OUT_PATH, p)


if __name__ == "__main__":
    main(force_download=True)
