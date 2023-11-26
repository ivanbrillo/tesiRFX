import pandas as pd
from scipy import signal
import numpy as np


def get_mov_avg(data_list: list, win_size: int) -> list:
    smoothed_list = list()
    for data_element in data_list:
        smoothed_list.append({
            "date": data_element["date"],
            "type": data_element["type"],
            "frequency": data_element["frequency"],
            "exposure time": data_element["exposure time"],
            "supply delay": data_element["supply delay"],
            "time_data": data_element["time_data"].rolling(window=win_size).mean()
        })

    return smoothed_list


def get_simple_exp(data_list: list, alpha: int) -> list:
    smoothed_list = list()
    for data_element in data_list:
        smoothed_list.append({
            "date": data_element["date"],
            "type": data_element["type"],
            "frequency": data_element["frequency"],
            "exposure time": data_element["exposure time"],
            "supply delay": data_element["supply delay"],
            "time_data": data_element["time_data"].ewm(alpha=alpha).mean()
        })

    return smoothed_list


def get_savitzy_golay(data_list: list, win_len: int, pol_order) -> list:
    smoothed_list = list()
    for data_element in data_list:
        smoothed_list.append({
            "date": data_element["date"],
            "type": data_element["type"],
            "frequency": data_element["frequency"],
            "exposure time": data_element["exposure time"],
            "supply delay": data_element["supply delay"],
            "time_data": pd.Series(signal.savgol_filter(data_element["time_data"], window_length=win_len, polyorder=pol_order, mode="nearest"))
        })

    return smoothed_list


def get_FFT(data_list: list, n_freq: int) -> list:
    smoothed_list = list()
    for data_element in data_list:
        rft = np.fft.rfft(data_element["time_data"].to_numpy())  # performs a real-valued Fast Fourier Transform (FFT), return array of frequency
        rft[n_freq:] = 0  # removes high-frequency components from the FFT. Note, rft.shape = 451
        smoothed_list.append({
            "date": data_element["date"],
            "type": data_element["type"],
            "frequency": data_element["frequency"],
            "exposure time": data_element["exposure time"],
            "supply delay": data_element["supply delay"],
            "time_data": pd.Series(np.fft.irfft(rft))  # performs the inverse FFT to get back in the time domain  
        })

    return smoothed_list
