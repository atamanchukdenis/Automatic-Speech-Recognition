import abc
from typing import List, Tuple
import numpy as np
import pandas as pd
from tensorflow import keras


class Dataset(keras.utils.Sequence):
    """
    The `Dataset` represents the sequence of samples used for Keras models.
    It has a view by the `reference` to sample sources, so we do not keep an
    entire dataset in the memory.

    The class contains two essential methods `len` and `getitem`, which are
    required to use the `keras.utils.Sequence` interface. This structure
    guarantee that the network only trains once on each sample per epoch.
    """

    def __init__(self,
                 references: pd.DataFrame,
                 batch_size: int):
        self._batch_size = batch_size
        self._references = references
        self._indices = np.arange(len(self))

    @property
    def indices(self):
        return self._indices

    def __len__(self) -> int:
        """ Indicate the number of batches per epoch. """
        return int(np.floor(len(self._references.index) / self._batch_size))

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        """ Get the batch data. We have an auxiliary index to have more
        control of the order, because basically model uses it sequentially. """
        aux_index = self._indices[index]
        return self.get_batch(aux_index)

    @abc.abstractmethod
    def get_batch(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        pass

    def shuffle_indices(self):
        """ Set up the order of return batches. """
        np.random.shuffle(self._indices)


class SortedDataset(Dataset):

    def __init__(self,
                 references: pd.DataFrame,
                 batch_size: int,
                 is_sorted: bool = False,
                 is_bins_behaviour: bool = False,
                 is_homogeneous: bool = False,
                 bins: int = 0):
        super().__init__(references, batch_size)

        self._is_sorted = is_sorted
        self._is_bins_behaviour = is_bins_behaviour
        self._is_homogeneous = is_homogeneous
        self._bins = bins

        if self._is_sorted or self._is_bins_behaviour:
            self.sort()

        if self._is_homogeneous:
            self.shuffle_homogeneous_bins()
        elif self._is_bins_behaviour:
            self.shuffle_bins()
        else:
            self.shuffle_indices()

        print('Dataset initialized:',
              self._references.iloc[0]['frames'],
              self._references.iloc[0 + self._batch_size]['frames'],
              self._references.iloc[self._references.shape[0] - 1 - self._batch_size]['frames'],
              self._references.iloc[self._references.shape[0] - 1]['frames'])

    def on_epoch_end(self):
        super().on_epoch_end()
        if self._is_homogeneous:
            self.shuffle_homogeneous_bins()
            print('homogeneous bins shuffled')
        elif self._is_bins_behaviour:
            self.shuffle_bins()
            print('bins shuffled')
        else:
            self.shuffle_indices()
            print('dataset shuffled')

    def sort(self):
        self._references = self._references \
            .sort_values(by=['frames']) \
            .reset_index(drop=True)

    def shuffle_bins(self):
        df_bins = np.array_split(
            self._references, self._bins)

        inds = np.arange(
            len(df_bins))
        np.random.shuffle(inds)
        inds = inds.tolist()

        df_bins_shuffled = []
        for ind in inds:
            df_bin = df_bins[ind]
            df_bins_shuffled.append(
                df_bin.sample(frac=1))
        df_bins = df_bins_shuffled

        self._references = pd \
            .concat(df_bins) \
            .reset_index(drop=True)

    def shuffle_homogeneous_bins(self):
        self.sort()

        frames = dict(
            [(f, 0) for f in list(
                np.arange(
                    self._references['frames'].min(),
                    self._references['frames'].max() + 1))])

        for i in range(
                self._references['frames'].min(),
                self._references['frames'].max() + 1):
            frames[i] = (self._references['frames'] == i).sum()

        frames = [frames[k] for k in frames.keys() if frames[k] != 0]

        frame_bounds = []
        insts = 0
        for f in frames:
            frame_bounds.append((insts, insts + f))
            insts += f

        df_bins_shuffled = []
        for fb in frame_bounds:
            df_bin = self._references[fb[0]:fb[1]]
            df_bins_shuffled.append(
                df_bin.sample(frac=1))
        df_bins = df_bins_shuffled

        self._references = pd \
            .concat(df_bins) \
            .reset_index(drop=True)