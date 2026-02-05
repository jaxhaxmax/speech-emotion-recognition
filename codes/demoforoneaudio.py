import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "Dataset/Actor_01/03-01-05-01-02-02-01.wav"

signal, sr = librosa.load(audio_path, sr=22050)

# extract MFCC
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=13
)

print("MFCC shape:", mfcc.shape)

# visualize MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis="time")
plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.show()
