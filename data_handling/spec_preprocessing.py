import librosa

def classifier_representation(y, window, step, sr, n_mels, fmin=400, fmax=12000):
    # Converting windows size and step size to nfft and hop length (in frames) because librosa uses that.
    n_fft = int(window * sr)  # Window size
    hop_length = int(step * sr)  # Step size
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window="hamming", n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec = librosa.power_to_db(S, ref=1.0, top_db=80.0)

    # representation_data = skimage.transform.resize(spec, (IMG_HEIGHT,IMG_WIDTH))
    # representation_data = scale_to_range(representation_data, -1, 1)

    return spec