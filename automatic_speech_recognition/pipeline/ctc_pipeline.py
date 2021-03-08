import os
from types import MethodType
import logging
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import Pipeline
from .. import augmentation
from .. import decoder
from .. import features
from .. import dataset
from .. import text
from .. import utils
from ..features import FeaturesExtractor
import kenlm
from collections import defaultdict, Counter
from string import ascii_lowercase
import re
import concurrent

model_arpa = kenlm.LanguageModel('/home/datamanc/data/CommonCrawl/400K_3-gram.arpa')

def lm(sent):
    return np.exp(
        model_arpa.score(sent))

def decode(lm,
           k,
           alpha,
           beta,
           prune,
           ctc,
           ctc_index):
    """
    Performs prefix beam search on the output of a CTC network.
    Args:
        ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
        lm (func): Language model function. Should take as input a string and output a probability.
        k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
        alpha (float): The language model weight. Should usually be between 0 and 1.
        beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
        prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.
    Retruns:
        string: The decoded CTC output.
    """

    lm = (lambda l: 1) if lm is None else lm # if no LM is provided, just set to function returning 1
    W = lambda l: re.findall(r'\w+', l)
    alphabet = [' '] + list(ascii_lowercase) + ['%']
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):

        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:

            """
            if len(l) > 0 and l[-1] == '>':
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue
            """

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2

                # STEP 3: “Extending” with a blank
                if c == '%':
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3

                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                    # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(' ', '')) > 0 and (c == ' ' or t == T - 1):
                        lm_prob = lm(l_plus.strip(' ')) ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7

    d = dict([(i, c) for c, i in enumerate(list(alphabet))])
    res = [d[c] for c in A_prev[0]]
    return np.array(res), ctc_index

def prefix_beam_search_decoder(batch_logits,
                               lm=None,
                               k=64,
                               alpha=0.1,
                               beta=4,
                               prune=0.001):
    res = []
    batch_size = batch_logits.shape[0]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(decode, lm, k, alpha, beta, prune, ctc, i) for i, ctc in enumerate(list(batch_logits))]
        r = concurrent.futures.wait(futures)

        for _ in range(batch_size):
            res.append(r.done.pop().result())

        res.sort(key = lambda x: x[1])
        res = [r for r, _ in res]

    return res

logger = logging.getLogger('asr.pipeline')


class CTCPipeline(Pipeline):
    """
    The pipeline is responsible for connecting a neural network model with
    all non-differential transformations (features extraction or decoding),
    and dependencies. Components are independent.
    """

    def __init__(self,
                 alphabet: text.Alphabet,
                 features_extractor: features.FeaturesExtractor,
                 model: keras.Model,
                 optimizer: keras.optimizers.Optimizer,
                 decoder: decoder.Decoder,
                 checkpoint_dir = None,
                 gpus: List[str] = None):
        self._alphabet = alphabet
        self._model_cpu = model
        self._optimizer = optimizer
        self._decoder = decoder
        self._features_extractor = features_extractor
        self._gpus = gpus
        self._checkpoint_dir = checkpoint_dir
        self._model = self.distribute_model(model, gpus) if gpus else model

    @property
    def alphabet(self) -> text.Alphabet:
        return self._alphabet

    @property
    def features_extractor(self) -> features.FeaturesExtractor:
        return self._features_extractor

    @property
    def model(self) -> keras.Model:
        return self._model_cpu

    @property
    def decoder(self) -> decoder.Decoder:
        return self._decoder

    def preprocess(self,
                   batch: Tuple[List[np.ndarray], List[str]],
                   is_extracted: bool,
                   augmentation: augmentation.Augmentation) -> Tuple[np.ndarray, np.ndarray]:
        """ Preprocess batch data to format understandable to a model. """
        data, transcripts = batch
        if is_extracted:  # then just align features
            features = FeaturesExtractor.align(data)
        else:
            features = self._features_extractor(data)
        features = augmentation(features) if augmentation else features
        labels = self._alphabet.get_batch_labels(transcripts)
        return features, labels

    def compile_model(self):
        """ The compiled model means the model configured for training. """
        y = keras.layers.Input(name='y', shape=[None], dtype='int32')
        loss = self.get_loss()
        self._model.compile(self._optimizer, loss, target_tensors=[y])
        logger.info("Model is successfully compiled")
        if self._checkpoint_dir is not None:
            self._model.load_weights(
                os.path.join(self._checkpoint_dir, 'model.h5'))
            logger.info("Model is loaded from",
                        os.path.join(self._checkpoint_dir, 'model.h5'))

    def fit(self,
            dataset: dataset.Dataset,
            dev_dataset: dataset.Dataset,
            augmentation: augmentation.Augmentation = None,
            prepared_features: bool = False,
            **kwargs) -> keras.callbacks.History:
        """ Get ready data, compile and train a model. """
        dataset = self.wrap_preprocess(dataset, prepared_features, augmentation)
        dev_dataset = self.wrap_preprocess(dev_dataset, prepared_features, augmentation)
        if not self._model.optimizer:  # a loss function and an optimizer
            self.compile_model()  # have to be set before the training
        return self._model.fit(dataset, validation_data=dev_dataset, **kwargs)

    def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
        """ Get ready features, and make a prediction. """
        features = self._features_extractor(batch_audio)
        batch_logits = self._model.predict(features, **kwargs)
        #decoded_labels = self._decoder(batch_logits)
        decoded_labels = prefix_beam_search_decoder(batch_logits, lm=lm)
        predictions = self._alphabet.get_batch_transcripts(decoded_labels)
        return predictions

    def wrap_preprocess(self,
                        dataset: dataset.Dataset,
                        is_extracted: bool,
                        augmentation: augmentation.Augmentation):
        """ Dataset does not know the feature extraction process by design.
        The Pipeline class exclusively understand dependencies between
        components. """
        def preprocess(get_batch):
            def get_prep_batch(index: int):
                batch = get_batch(index)
                return self.preprocess(batch, is_extracted, augmentation)
            return get_prep_batch
        dataset.get_batch = preprocess(dataset.get_batch)
        return dataset

    def save(self, directory: str):
        """ Save each component of the CTC pipeline. """
        self._model.save(os.path.join(directory, 'model.h5'))
        utils.save(self._alphabet, os.path.join(directory, 'alphabet.bin'))
        utils.save(self._decoder, os.path.join(directory, 'decoder.bin'))
        utils.save(self._features_extractor,
                   os.path.join(directory, 'feature_extractor.bin'))

    @classmethod
    def load(cls, directory: str, **kwargs):
        """ Load each component of the CTC pipeline. """
        model = keras.model.load_model(os.path.join(directory, 'model.h5'))
        alphabet = utils.load(os.path.join(directory, 'alphabet.bin'))
        decoder = utils.load(os.path.join(directory, 'decoder.bin'))
        features_extractor = utils.load(
            os.path.join(directory, 'feature_extractor.bin'))
        return cls(alphabet, model, model.optimizer, decoder,
                   features_extractor, **kwargs)

    @staticmethod
    def distribute_model(model: keras.Model, gpus: List[str]) -> keras.Model:
        """ Replicates a model on different GPUs. """
        try:
            dist_model = keras.utils.multi_gpu_model(model, len(gpus))
            logger.info("Training using multiple GPUs")
        except ValueError:
            dist_model = model
            logger.info("Training using single GPU or CPU")
        return dist_model

    @staticmethod
    def get_loss() -> Callable:
        """ The CTC loss using TensorFlow's `ctc_loss`. """
        def get_length(tensor):
            lengths = tf.math.reduce_sum(tf.ones_like(tensor), 1)
            return tf.cast(lengths, tf.int32)

        def ctc_loss(labels, logits):
            label_length = get_length(labels)
            logit_length = get_length(tf.math.reduce_max(logits, 2))
            return tf.nn.ctc_loss(labels, logits, label_length, logit_length,
                                  logits_time_major=False, blank_index=-1)
        return ctc_loss
