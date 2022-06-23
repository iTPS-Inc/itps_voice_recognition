#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import torchaudio
from fastdownload import FastDownload
from fastai.data.all import Path, get_files, untar_data

DATAROOT = os.environ.get("PREPROCESS_DATAROOT")
URL="https://www.dropbox.com/s/qzvrx0c3rrmxxl3/annotation_data_initial.tar.gz?dl=1"

d = FastDownload(base=DATAROOT)
DATAROOT = d.get(URL, force=False)


audio_files = get_files(DATAROOT, extensions=[".mp4"])
if os.path.exists(DATAROOT/"annotation_data.csv"):
    os.unlink(DATAROOT / "annotation_data.csv")
outdir = DATAROOT.parent / "annotation_data_out"
if not os.path.exists(outdir):
    os.mkdir(outdir)

print(outdir)


def convert_to_wav(input_file, output_file):
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{input_file}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            f"{output_file}",
        ]
    )
    return output_file


for audio_file in audio_files:
    if not os.path.exists(outdir / "base_wavs"):
        os.mkdir(outdir / "base_wavs")
    if "English" in str(audio_file):
        prep = "en"
    elif "Japanese" in str(audio_file):
        prep = "jp"
    else:
        prep = "_"
    print(audio_file)
    out_file_path = outdir / "base_wavs" / (f"{prep}_" + audio_file.stem)
    out_file_path = "_".join(str(out_file_path).split(" ")) + ".wav"
    convert_to_wav(audio_file, out_file_path)


# These are the files that I don't think I received the files for
#   '05_20190405_TP_Discussion_Internal _Meeting.mp4',
# # '08_20200813_Pokemon_pwc_call_APA.m4a',
# # '09_20201003_PokémoncallPWC_GT_Tsutsui-san.m4a',
# # '10_20201017_PokémonPWC_GT_Tsutsui-san.m4a'
# # '11_20200110_APA_prior_consultation.m4a',
# # '12_20201030 APA walkthrough RPSM rev.m4a',
# # '13_20201029 APA walkthrough CPS rev.m4a',
#   "[OECD_Tax]_Concept_of_Transfer_Pricing_and_Arm’s_Length_Principle_Lecture_1-_Kyung_Geun_Lee.wav": ,
#   "Transfer_pricing_implications_in_a_post-BEPS_and_post-US_tax_reform_environment.wav": ,
#   "Baker_Tilly_International_-_International_Corporate_Taxation_-_BEPS_Action_Plan_Overview.wav"  : ,

base_excel_translations = {
    "en_Transfer_Pricing_2015.wav": "04_Transfer_Pricing_2015.mp4",
    "en_OECD_Tax_Talks_#14_-_January_2020_-_Centre_for_Tax_Policy_and_Administration.wav": "08_OECD_Tax_Talks#14-January_2020-Centre_for_Tax_Policy_and_Administration",
    "en_Transfer_Pricing_and_Tax_Avoidance.wav": "01_Transfer_Pricing_and_Tax_Avoidance.mp4",
    "en_Transfer_pricing_and_tax_havens__Taxes__Finance_&_Capital_Markets__Khan_Academy.wav": "02_Transfer_pricing_and_tax_havens_Taxes_Finance_Capital_Markets_Khan_Academy.mp4",
    "jp_03_移転価格の基礎.wav": "01_03_Transfer_Pricing_Basics.mp4",
    "jp_42号公告による移転価格コンプライアンスへの影響（中国）.wav": "04_No42_announcement_transprice_complience_PRC.mp4",
    "jp_BEPS：移転価格文書化新規定の概要および日本企業の課題・対策.wav": "06_BEPS_transprice_documentation_regulation_issues.mp4",
    "jp_ジェトロ税務WEBセミナー「パンデミックでの移転価格対応（米国）」.wav": "07_Jetro_taxation_webiner_pandemic_transprice_issues_USA.mp4",
    "jp_移転価格についてわかりやすく解説します.wav": "02_Easy_Explanation_of_Transfer_Pricing.mp4",
    "jp_移転価格対応講座101（20200928実施）※視聴は社員のみ.wav": "03_Lesson101_Transfer_Pricing_20200928.mp4",
}
excel_trans = {v: k for k, v in base_excel_translations.items()}

csv_files = get_files(DATAROOT, extensions=[".csv"])
df = pd.concat(
    [pd.read_csv(csv, low_memory=False) for csv in csv_files], ignore_index=True
)
df = df[df["Source Video Name"].notna()]
df["wav_file_name"] = df["Source Video Name"].apply(lambda x: excel_trans.get(x, None))
df = df[df["wav_file_name"].notna()].reset_index(drop=True)

df[["st", "text", "et"]] = df["Transcription"].str.extract(r"(\[.*\])(.*)(\[.*\])")
df["st"] = df["st"].str.extract(r"(\[[\d\.:]+\])(\[.*\])*")[0]


def convert_time(time):
    # 61:32.117
    if len(time) == 9:
        minutes = int(time[:2])
        secs = time[3:]
        hours, mins = minutes // 60, minutes % 60
        return f"{hours}:{mins}:{secs}"
    elif len(time) == 11:
        return time


def cut_out_part_ffmpeg(inp, st, end, out):
    st, end = st[1:-1], end[1:-1]
    st, end = convert_time(st), convert_time(end)

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{inp}",
            "-ss",
            f"{st}",
            "-to",
            f"{end}",
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


df["test"] = False
df.loc[df.sample(frac=0.2).index, "test"] = True
df["test_string"] = df["test"].apply(lambda x: "test" if x else "train")
df["base_path"] = df["test_string"].apply(lambda _: str(outdir / "wavs"))

os.mkdir(outdir / "wavs")
os.mkdir(outdir / "wavs" / "train")
os.mkdir(outdir / "wavs" / "test")

outfile_names = []
for i, wav, test_string, st, end in df[["wav_file_name", "test_string", "st", "et"]].itertuples():

    infile = outdir / "base_wavs" / wav
    out_name = f"wavs/{test_string}/" + str(Path(wav).stem + f"_{i}.wav")
    outfile = outdir / out_name
    outfile_names.append(out_name)

    cut_out_part_ffmpeg(infile, st, end, outfile)

df["file"] =  outfile_names
no_frames = []
srs = []
for f in outfile_names:
    t, sr = torchaudio.load(str(outdir / f))
    no_frames.append(len(t.squeeze()))
    srs.append(sr)


df["sr"] = srs
df["no_frames"] = no_frames
df["audio_length"] = df["no_frames"] / df["sr"]
df["lang"] = df["wav_file_name"].apply(lambda x: x[:2])

# Get the relative path
df_out = df[["Transcription", "Keywords", "Comment",
         "wav_file_name", "st", "text", "et", "test", "file",
         "sr", "audio_length", "no_frames"
         ]].copy()

curdir= os.getcwd()
os.chdir(DATAROOT.parent)
df_out.to_csv(outdir / "df.csv")
subprocess.run(
    [
        "zip",
        "-r",
        f"annotation_data.zip",
        f"annotation_data_out"
    ]
)
os.chdir(curdir)
