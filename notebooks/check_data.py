#!/usr/bin/env python3
import functools
import torchaudio
@functools.lru_cache(maxsize=None)
def can_load(f):
  try:
    torchaudio.load(f)
    return True
  except Exception as e:
    return False

def save_new_frame_with_correct_files(df, lang):
  df["can_load"] = df["corrected_filename"].apply(can_load)
  df[df["can_load"]].to_csv(audio_path / ("full_can_load_"+lang+".csv"))
  couldnt_load = df.shape[0] - df["can_load"].sum()
  df = df[df["can_load"]]
  df[df["audio_filename"].str.contains("test")].to_csv(audio_path/("test_can_load_"+lang+".csv"))
  df[df["audio_filename"].str.contains("train")].to_csv(audio_path/("train_can_load_"+lang+".csv"))
  print("couldnt load:", couldnt_load)
