import pandas as pd
from scipy import signal
import numpy as np


def _create_dict(original_list: list, smoothed_data: list) -> list:
    data_list = list()
    for data_dict, smoothed_list in zip(original_list, smoothed_data):
        data_list.append({
            "date": data_dict["date"],
            "type": data_dict["type"],
            "frequency": data_dict["frequency"],
            "exposure time": data_dict["exposure time"],
            "supply delay": data_dict["supply delay"],
            "time_data": smoothed_list
        })
    return data_list


def get_mov_avg(data_list: list, win_size: int) -> list:
    smoothed_data = [data["time_data"].rolling(window=win_size).mean() for data in data_list]
    return _create_dict(data_list, smoothed_data)


def get_simple_exp(data_list: list, alpha: int) -> list:
    smoothed_data = [data["time_data"].ewm(alpha=alpha).mean() for data in data_list]
    return _create_dict(data_list, smoothed_data)


def get_savitzy_golay(data_list: list, win_len: int, pol_order) -> list:
    smoothed_data = [pd.Series(
        signal.savgol_filter(data["time_data"], window_length=win_len, polyorder=pol_order,
                             mode="nearest")) for data in data_list]
    return _create_dict(data_list, smoothed_data)


def get_FFT(data_list: list, n_freq: int) -> list:
    rft_list = list()
    for data_element in data_list:
        # performs a real-valued Fast Fourier Transform (FFT), return array of frequency
        rft_list.append(np.fft.rfft(data_element["time_data"].to_numpy()))
        rft_list[-1][n_freq:] = 0  # removes high-frequency components from the FFT. Note, rft.shape = 451

    # performs the inverse FFT to get back in the time domain
    smoothed_data = [pd.Series(np.fft.irfft(rft)) for rft in rft_list]
    return _create_dict(data_list, smoothed_data)


def _atmf(data: list, win_size: int, alpha: int) -> pd.Series:
    N = len(data)
    result = [0] * (N - win_size)

    assert win_size > alpha > 0 and alpha % 4 == 0 and win_size % 2 == 0

    for i in range(win_size // 2, N - win_size // 2):
        window = data[i - win_size // 2:i + win_size // 2]
        window.sort()

        # Get the result - the mean value of the elements in the trimmed set
        result[i - win_size // 2] = sum(window[alpha // 4:win_size - (alpha // 4) * 3]) / (win_size - alpha)

    return pd.Series([result[0]] * (win_size // 2) + result + [result[-1]] * (win_size // 2))


def alpha_trimmed_mean_filter(data_list: list, win_size: int, alpha: int) -> list:
    smoothed_data = [_atmf(data["time_data"].tolist(), win_size, alpha) for data in data_list]
    return _create_dict(data_list, smoothed_data)
