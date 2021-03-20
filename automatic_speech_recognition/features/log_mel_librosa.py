import numpy as np
import librosa
from .. import features

class LogMelLibrosa(features.FeaturesExtractor):

    def __init__(self, features_num: int, is_standardization=True, **kwargs):
        self.features_num = features_num
        self.is_standardization = is_standardization
        self.params = kwargs
        self.sr = 16000
        self.n_fft = int(self.sr / (1 / self.params['winlen']))
        self.hop_length = int(self.sr / (1 / self.params['winstep']))
        self.winfunc = self.params['winfunc']

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        melSpectrum = librosa.feature.melspectrogram(
            audio.astype(np.float16),
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.features_num,
            window=self.winfunc)
        logMelSpectrogram = librosa.power_to_db(
            melSpectrum, ref=np.max)
        features = logMelSpectrogram.T
        return self.standardize(features) if self.is_standardization else features