#!/usr/bin/env python3
from fastai.data.all import *
import os
import pandas as pd
import subprocess
import srt
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
import torchaudio
import shutil
import uuid

tqdm.pandas()

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(filename="info_file.log"))
logger.addHandler(logging.StreamHandler())
logger.info("test logging")

df = pd.read_csv(
    "other_data.csv",
    sep="\t",
)

df = df.set_index("name").to_dict("index")
PATH = Path("/Volumes/Elements/audio_data_prep/video/")
OUTPATH = Path("/Volumes/Elements/audio_data_prep/out/other_data/")
if not os.path.exists(OUTPATH):
    OUTPATH.parent.mkdir(exist_ok=True)
    OUTPATH.mkdir()


WORKDIR = Path().home() / "tmp" / "workdir_1"
WORKDIR.mkdir(exist_ok=True)

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

def get_subtitles(subtitle_file):
    with open(subtitle_file, "r") as f:
        subtitles_raw = f.read()
        subtitles = srt.parse(subtitles_raw)
    return list(subtitles)


def cut_clip_from_subtitle(subtitle, video_file, out_name):
    inp = video_file
    if subtitle.start > datetime.timedelta(milliseconds=500):
        start = str(subtitle.start - datetime.timedelta(milliseconds=500))
    else:
        start = str(subtitle.start)
    end = str((subtitle.end - subtitle.start) + datetime.timedelta(milliseconds=500))
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
        logger.warning(f"Failed to cut out subtitles {subtitle} at {start}:{end} from {inp} to {out_name}")
    return x


srt_files = get_files(PATH / g["subtitle_folder"], extensions=[".srt"])
video_files = get_files(PATH /g["video_folder"], extensions=[".mkv"])

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

def do_subtitle_file(subtitle, subtitle_file, video):
    out_file_id = uuid.uuid4()
    out_file = WORKDIR / (f"{ out_file_id }" + ".wav")
    if not os.path.exists(out_file):
        _ = cut_clip_from_subtitle(out_name=out_file, subtitle=subtitle, video_file=video)
    logger.info("Subtitle file,video fil e,contents,outfile")
    logger.info(f"{subtitle_file},{video},{subtitle.content},{out_file}")
    return [subtitle_file, video, subtitle.content.replace("\n", " "), out_file]


def find_video(s):
    return clip_frame[clip_frame["file_names"].apply(lambda x: x.split("/")[-1].startswith(s))]

def process_video(video):
    subtitle_file = file_to_subtitle.get(video, None)
    if not subtitle_file:
        return None
    out_list = Parallel(n_jobs=-1)(delayed(do_subtitle_file)(subtitle, subtitle_file, video) for subtitle in get_subtitles(subtitle_file))
    return out_list

results = Parallel(n_jobs=-1)(delayed(process_video)(video_file) for video_file in tqdm(video_files))

out=[]
for res in results:
    if res:
        out+=res

clip_frame = pd.DataFrame(out, columns=["subtitle_file", "video", "content", "file"])
clip_frame["test"] = False
clip_frame.loc[clip_frame.sample(frac=0.2).index, "test"] = True
clip_frame["train"]  = ~clip_frame["test"]

def try_loading(s):
    try:
        t, sr = torchaudio.load(s)
        no_frames = t.squeeze().shape[0]
        return pd.Series((no_frames, sr))
    except Exception as e:
        logger.debug(e)
        return pd.Series((None, None))

clip_frame[["no_frames", "sr"]] = clip_frame["file"].progress_apply(try_loading)
clip_frame["audio_length"] = clip_frame["no_frames"] / clip_frame["sr"]

clip_frame = clip_frame[clip_frame["no_frames"] > 0]
clip_frame["split"] = clip_frame["test"].apply(lambda x: "test" if x else "train")

def move_file(s):
    after_move_name = s["file"].name.strip()
    rel_path = s["split"] + "/" + after_move_name
    if not os.path.exists(OUTPATH / s["split"]):
        (OUTPATH / s["split"]).mkdir()
    if os.path.exists(s["file"]):
        shutil.copy(s["file"], OUTPATH / rel_path)
    else:
        return None
    return rel_path

clip_frame["file_names"] = clip_frame[["file", "split"]].progress_apply(move_file, axis=1)

curdir = os.getcwd()
os.chdir(OUTPATH.parent)
clip_frame.to_csv(OUTPATH / "clip_frame.csv")
subprocess.run(["tar", "-zcvf", "other_data.zip", "other_data"])
os.chdir(curdir)
