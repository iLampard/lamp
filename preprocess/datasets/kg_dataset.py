import math

import numpy as np
from torch.utils.data import Dataset


class KGDataset(Dataset):
    """Return a sequence whose length is context_length + 1, and drop last by default"""

    def __init__(
            self,
            data,
            context_length: int,
            time_factor: float = 100.0,
            start_ratio: float = 0.0,
            end_ratio: float = 1.0,
            overlap: bool = True,
    ):
        # data is the list of tuple
        self.data = data
        self.context_length = context_length
        self.time_factor = time_factor
        # start and end ratio of the length of the data
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.overlap = overlap

        self.window_length = self.context_length + 1

        # set start and end indexes for each sequence (just one sequence in this case)
        data_length = len(self.data)
        assert data_length >= self.window_length, f'The length of dataset ({data_length}) ' \
                                                  f'is less than context_length ({self.window_length})'

        self.start_idx = int(len(self.data) * self.start_ratio)
        self.end_idx = int(len(self.data) * self.end_ratio)

    def __getitem__(self, idx):
        """Get a sequence of data whose length is context_length + 1 ([idx - context_length, idx])

        Args:
            idx:

        Returns:

        """
        original_idx = self.get_original_index(idx)
        sample = {
            'seq_subject': self._extract_window_list_by_idx(original_idx, tuple_idx=0),
            'seq_object': self._extract_window_list_by_idx(original_idx, tuple_idx=1),
            'seq_relation': self._extract_window_list_by_idx(original_idx, tuple_idx=2),
            'seq_time': (self._extract_window_list_by_idx(original_idx, tuple_idx=3) / self.time_factor).astype(
                np.float32),
        }
        return sample

    def __len__(self):
        if self.overlap:
            num = self.end_idx - self.start_idx - self.context_length
        else:
            num = math.floor((self.end_idx - self.start_idx) / self.window_length)
        return num

    def get_original_index(self, idx):
        if self.overlap:
            return self.start_idx + self.context_length + idx
        else:
            return self.start_idx + self.context_length + idx * self.window_length

    def _extract_window_list_by_idx(self, idx, tuple_idx):
        window = []
        for i in range(idx - self.context_length, idx + 1):
            window.append(self.data[i][tuple_idx])
        return np.array(window)
