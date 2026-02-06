from features import extract_mfcc_fixed

path = r"C:\Users\tanma\OneDrive\Desktop\speech-emotion-recognition\Dataset\Actor_01\03-01-05-01-02-02-01.wav"

mfcc = extract_mfcc_fixed(path)
print(mfcc.shape)
