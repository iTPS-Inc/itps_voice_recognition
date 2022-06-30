#!/usr/bin/env python3
from fastai.data.all import *
import os
import pandas as pd
import tempfile
import subprocess
import srt
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import logging

logger = logging.getLogger("video processor")
logger.setLevel(logging.DEBUG)


df = pd.read_csv(
    "other_data.csv",
    sep="\t",
)

df = df.set_index("name").to_dict("index")
PATH = Path("/run/media/jjs/Elements/audio_data_prep/video/")
subtitles = Path("/run/media/jjs/Elements/audio_data_prep/video/kitsunekko_backup/")
WORKDIR = Path(tempfile.gettempdir()) / "workdir_1"
WORKDIR.mkdir(exist_ok=True)

srt_files = get_files(subtitles, extensions=[".srt"])
g = df["Gintama"]

episode_number_re = re.compile(".* - (\d+) \[")


def get_episode_number(video_file):
    name = video_file.name
    res = episode_number_re.search(name)
    if res:
        episode_number = res[1]
        return episode_number
    else:
        return None


def match_gintama(subtitle_file, video_files):
    name = subtitle_file.stem
    number = name[-3:]
    srt_files[0]

def extract_subtitles(inp, out):
    x = subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{inp}",
            "-ss",
            f"{st_new}",
            "-to",
            f"{end_new}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            f"{out}",
            "-y",
        ]
    )
    if not x.returncode == 0:
        l.log(st, end, st_new, end_new)
    return x


def get_subtitles(subtitle_file):
    with open(subtitle_file, "r") as f:
        subtitles_raw = f.read()
        subtitles = srt.parse(subtitles_raw)
    return subtitles


def cut_clip_from_subtitle(subtitle, video_file, out_name):
    inp = video_file
    if subtitle.start > datetime.timedelta(milliseconds=2000):
        start = str(subtitle.start - datetime.timedelta(milliseconds=2000))
    else:
        start = str(subtitle)
    end = str((subtitle.end - subtitle.start) + datetime.timedelta(milliseconds=2000))
    x = subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{inp}",
            "-ss",
            f"{start}",
            "-t",
            f"{end}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            f"{out_name}",
            "-y",
        ] ,
        stderr=subprocess.STDOUT,
        stdout=subprocess.DEVNULL

    )
    if not x.returncode == 0:
        logger.log(msg=f"start at {start}", level=logging.DEBUG)
        logger.log(msg=f"end at {end}", level=logging.DEBUG)
    return x


srt_files = get_files(g["subtitle_folder"], extensions=[".srt"])
video_files = get_files(g["video_folder"], extensions=[".mkv"])

def episode_number(s):
    res = episode_number_re.search(str(s))
    if res:
        return res[1]
    else:
        return None

subtitle_to_no = dict(zip(srt_files, srt_files.map(episode_number)))
no_to_file = dict(zip(video_files.map(get_episode_number), video_files))
subtitle_to_file = {
    k: no_to_file.get(v) for k, v in subtitle_to_no.items() if v in no_to_file
}

file_to_subtitle = {v: k for k, v in subtitle_to_file.items()}

super_out_list = []

def process_video(video):
    subtitle_file = file_to_subtitle.get(video, None)
    if not subtitle_file:
        return None
    out_list =[]
    for i, subtitle in enumerate(get_subtitles(subtitle_file)):
        out_file = WORKDIR / (f"{i:012d}" + ".wav")
        out = cut_clip_from_subtitle(out_name=out_file,
                                    subtitle=subtitle,
                                     video_file=video)
        logger.info("Subtitle file,video file,contents,outfile")
        logger.info(f"{subtitle_file},{video},{subtitle.content},{out_file}")
        out_list.append([subtitle_file, video, subtitle.content, out_file])
    return out_list


results = Parallel(n_jobs=28)(delayed(process_video)(video_file) for video_file in tqdm(video_files))

# for video_file in  video_files:

#     subtitle_file = file_to_subtitle.get(video_file, None)
#     if not subtitle_file:
#         continue
#     out_list =[]
#     for i, subtitle in enumerate(get_subtitles(subtitle_file)):
#         out_file = WORKDIR / (f"{i:012d}" + ".wav")
#         out = cut_clip_from_subtitle(out_name=out_file,
#                                     subtitle=subtitle,
#                                      video_file=video_file)
#         print("Subtitle file,video file,contents,outfile")
#         print(f"{subtitle_file},{video_file},{subtitle.content},{out_file}")
#         out_list.append([subtitle_file, video_file, subtitle.content, out_file])
#     super_out_list += out_list
