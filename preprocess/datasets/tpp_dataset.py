from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset


# ref: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/data/NHPDataset.py


class TPPDataset(Dataset):
    def __init__(
            self,
            data,
            num_event_types,
            start_date: str = '2000-01-01',
            end_date: str = '2030-01-01',
            time_factor: float = 100.0
    ):
        self.data = data
        self.time_factor = time_factor
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date

        # only use end date to do the truncation
        # because we better need the prefix sequence in valid and test set
        self.time_seqs_used_in_model = [
            [x["event_time"] for x in seq if self.is_before_time_period(x['event_date'])] for
            k, seq in data.items()]

        self.time_seqs_in_period = [
            [x["event_time"] for x in seq if self.is_in_time_period(x['event_date'])] for
            k, seq in data.items()]

        self.type_seqs = [
            [x["event_type"] for x in seq if self.is_before_time_period(x['event_date'])] for
            k, seq in data.items()]

        self.time_delta_seqs = [[x["event_dtime"] for x in seq if self.is_before_time_period(x['event_date'])]
                                for k, seq in data.items()]

        # make the first timestamp to zero
        self.time_seqs = []
        for seq in self.time_seqs_used_in_model:
            seq = [x - seq[0] for x in seq]
            self.time_seqs.append(seq)

        # seq idx
        self.seq_idx = list(data.keys())

        # position in the seq
        self.original_idx = self.get_original_index()

        self.event_num = num_event_types
        self.pad_index = self.event_num

    def get_original_index(self):
        res = []
        for time_seq, source_seq in zip(self.time_seqs_in_period, self.time_seqs_used_in_model):
            idx_in_seq = [source_seq.index(time_seq[i]) for i in range(len(time_seq))]
            res.append(idx_in_seq)
        return res

    def is_in_time_period(self, event_date: str, date_format='%Y-%m-%d'):
        event_date_ = datetime.strptime(event_date, date_format)
        if event_date_ >= self.start_date and event_date_ < self.end_date:
            return True
        else:
            return False

    def is_before_time_period(self, event_date: str, date_format='%Y-%m-%d'):
        event_date_ = datetime.strptime(event_date, date_format)
        if event_date_ < self.end_date:
            return True
        else:
            return False

    def __len__(self):
        """

        Returns: length of the dataset

        """

        return len(self.time_seqs)

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            time_seq, time_delta_seq and event_seq element

        """
        return self.time_seqs[idx], self.time_delta_seqs[idx], self.type_seqs[idx], self.seq_idx[idx], \
               self.original_idx[idx]

    def padding(self, seqs, dtype, max_len=None, pad_index=None):
        pad_index = self.pad_index if pad_index is None else pad_index
        # padding to the max_length
        if max_len is None:
            max_len = max(len(seq) for seq in seqs)
        batch_seq = np.array([seq + [pad_index] * (max_len - len(seq)) for seq in seqs], dtype=np.float64)

        return torch.tensor(batch_seq, dtype=dtype)

    def createPadAttnMask(self, type_seqs, concurrent_mask=None):
        # 1 -- pad, 0 -- non-pad
        batch_size, seq_len = type_seqs.size(0), type_seqs.size(1)
        batch_seq_pad_mask = type_seqs.eq(self.pad_index)
        attention_key_pad_mask = batch_seq_pad_mask.unsqueeze(1).expand(batch_size, seq_len, -1)
        subsequent_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=type_seqs.device, dtype=torch.uint8), diagonal=0
        ).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = subsequent_mask | attention_key_pad_mask.bool()
        if concurrent_mask is None:
            # no way to judge concurrent events, simply believe there is no concurrent events
            pass
        else:
            attention_mask |= concurrent_mask.bool()
        return ~batch_seq_pad_mask, attention_mask

    def collate_fn(self, batch):
        """

        Args:
            batch: batch sequence data

        Returns:
            batch tensors of time_seqs, time_delta_seqs, event_seqs,
            batch_non_pad_mask, attention_mask, type_mask

        """
        time_seqs, time_delta_seqs, type_seqs, seq_idx, original_idx = list(zip(*batch))

        # one could use float64 to avoid precision loss during conversion from numpy.array to torch.tensor
        # for generality we use float32 for the moment
        time_seqs = self.padding(time_seqs, torch.float32)
        time_delta_seqs = self.padding(time_delta_seqs, torch.float32)
        type_seqs = self.padding(type_seqs, torch.long)

        batch_non_pad_mask, attention_mask = self.createPadAttnMask(type_seqs)

        type_mask = torch.zeros([*type_seqs.size(), self.event_num])
        for i in range(self.event_num):
            type_mask[:, :, i] = type_seqs == i

        # an ugly pad, we fix it later
        # this pad has no effect in evaluation because batch_size
        original_idx = self.padding(original_idx, pad_index=1000, dtype=torch.long)

        seq_idx = np.array(seq_idx)[:, None]
        seq_idx = torch.LongTensor(np.tile(seq_idx, [1, original_idx.size(-1)]))

        return time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask, \
               seq_idx, original_idx
