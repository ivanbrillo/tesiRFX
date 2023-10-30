import tesiRFX.plot_drawer as plot_drawer   # custom module
import pandas as pd
from scipy import signal
import numpy as np


def get_mov_avg(df: pd.DataFrame, win_size: int) -> pd.DataFrame:
  column_names = plot_drawer.get_columns(df)  # get all the column with the exclusion of TIME
  smoothed_df = pd.DataFrame(df["TIME"])

  for name in column_names:
    smoothed_df[name] = df[name].rolling(window=win_size).mean()

  return smoothed_df


def get_simple_exp(df: pd.DataFrame, alpha: int) -> pd.DataFrame:
  column_names = plot_drawer.get_columns(df)
  smoothed_df = pd.DataFrame(df["TIME"])

  for name in column_names:
    smoothed_df[name] = df[name].ewm(alpha=alpha).mean()
  
  return smoothed_df


def get_savitzy_golay(df: pd.DataFrame, win_len: int, pol_order) -> pd.DataFrame:
  column_names = plot_drawer.get_columns(df)
  smoothed_df = pd.DataFrame(df["TIME"])

  for name in column_names:
    smoothed_df[name] = signal.savgol_filter(df[name], window_length=win_len, polyorder=pol_order, mode="nearest")

  return smoothed_df


def get_FFT(df: pd.DataFrame, n_freq: int) -> pd.DataFrame:
  column_names = plot_drawer.get_columns(df)
  smoothed_df = pd.DataFrame(df["TIME"])

  for name in column_names:
    rft = np.fft.rfft(df[name].to_numpy())   # performs a real-valued Fast Fourier Transform (FFT), return array of frequency
    rft[n_freq:] = 0                         # removes high-frequency components from the FFT. Note, rft.shape = 451
    smoothed_df[name] = np.fft.irfft(rft)    # performs the inverse FFT to get back in the time domain

  return smoothed_df
