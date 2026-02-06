print("features.py is being imported")

import numpy as np
import librosa


def extract_mfcc_fixed(
    audio_path,
    n_mfcc=13,
    max_len=200
):
    signal, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc
