#!/usr/bin/env python3
from fastai.data.all import untar_data, get_files
import pandas as pd
import shutil, os

ANNOTATION_DATA_URL_DROPBOX = (
    "https://www.dropbox.com/s/92tbhcdlymk5s0w/annotation_data.tar.gz?dl=1"
)


def __main__():
    p = untar_data(ANNOTATION_DATA_URL_DROPBOX)
    dfs = []
    for lang in ["en", "jp"]:
        wav_files = get_files(p / lang, extensions=[".wav"])
        csv_files = get_files(p / lang, extensions=[".csv"])
        df = pd.concat([pd.read_csv(i) for i in csv_files]).reset_index()
        df[["st", "text", "et"]] = df["Transcription"].str.extract(
            r"(\[.*\])(.*)(\[.*\])"
        )
        # df["filename"] = df["filename"].apply(lambda x: "/".join(str(x).split("/train/")))
        # df["exists"] = df["filename"].apply(os.path.exists)
        # wav_files = get_files(p, extensions=[".wav"])
        a = {}
        for i in wav_files:
            a[i.name] = i
        df["filename"] = df["Segmented audio file name"].apply(lambda x: a[x])
        df["lang"] = lang
        train = df.sample(frac=0.8).index
        df.loc[:, "train"] = False
        df.loc[train, "train"] = True
        df["test"] = ~df["train"]
        for ind, r in tqdm(df.iterrows()):
            if r["test"]:
                src = r["filename"]
                langfolder = src.parent.parent
                new_dir = langfolder / "test" / src.name
                if not os.path.exists(new_dir.parent):
                    os.mkdir(new_dir.parent)
                if os.path.exists(new_dir) and not os.path.exists(src):
                    continue
                shutil.move(src, new_dir)
            if not r["test"]:
                src = r["filename"]
                langfolder = src.parent.parent
                new_dir = langfolder / "train" / src.name
                if not os.path.exists(new_dir.parent):
                    os.mkdir(new_dir.parent)
                if os.path.exists(new_dir) and not os.path.exists(src):
                    continue
                shutil.move(src, new_dir)
        dfs = dfs + [df]

    df = pd.concat(dfs).drop("index", axis="columns").reset_index(drop=True)
    df["dataset"] = df["train"].apply(lambda x: "train" if x else "test")
    df["filename"] = df.apply(
        lambda x: x["filename"].parent.parent / x["dataset"] / x["filename"].name,
        axis=1,
    )
    df.to_csv(p / "annotation_data.csv")
