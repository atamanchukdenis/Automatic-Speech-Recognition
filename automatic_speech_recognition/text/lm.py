import numpy as np
import kenlm

class LanguageModel:

    def __init__(self,
                 path):
        self.path = path

    def load(self):
        ken_model = kenlm.LanguageModel(
            self.path)
        def lm(sent):
            return np.exp(
                ken_model.score(sent))
        return lm