import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ref: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/data/NHPDataset.py


class TPPDataset(Dataset):
    def __init__(self, data, event_num, eos_elapse=1, add_bos=True, add_eos=True, eps=np.finfo(float).eps,
                 concurrent=False,
                 skip_padding=False):
        """
        :param data: list[list[dict{"time_since_last_event"[float], "time_since_start"[float], "type_event"[int]}]]
        :param event_num: int, how many events are there in the whole dataset? (not limited to train/test/dev)
        each internal list is a event stream sequence
        following the data format of GaTech and JHU
            event_type(int) : starting from 0, appended [BOS], [EOS] and [PAD] at |E|, |E| + 1, |E| + 2
        :param eos_elapse: int, how much time we should wait after the last event to give EOS mark
        :param add_bos / add_eos: bool, whether to add [BOS] / [EOS]
        :param eps: float, if |x1-x2| < eps, then x1 == x2 (avoid float error in comparison)
        :param concurrent: bool, whether to consider concurrent events or not
        """
        assert eos_elapse >= 0, "EOS should not appear earlier than last event!"
        self.time_seq = [[x["time_since_start"] for x in seq] for seq in data]
        self.event_seq = [[x["type_event"] for x in seq] for seq in data]
        self.time_delta_seq = [[x["time_since_last_event"] for x in seq] for seq in data]
        self.skip_padding = skip_padding

        # starting from 0
        self.event_num = event_num
        self.pad_index = self.event_num
        self.bos_index = self.event_num + 1
        self.eos_index = self.event_num + 2
        self.eps = eps
        self.concurrent = concurrent

        self.add_bos = add_bos
        self.add_eos = add_eos
        self.data = data
        # at least include [PAD]
        self.num_types = self.event_num + 1
        assert max(
            [max(seq) for seq in self.event_seq]) + 1 <= self.event_num, "there are more event than specified?"
        if self.add_bos:
            self.time_seq = [[0, ] + seq for seq in self.time_seq]
            self.event_seq = [[self.bos_index, ] + seq for seq in self.event_seq]
            self.time_delta_seq = [[0, ] + seq for seq in self.time_delta_seq]
            self.num_types += 1
        if self.add_eos:
            self.time_seq = [seq + [seq[-1] + eos_elapse, ] for seq in self.time_seq]
            self.event_seq = [seq + [self.eos_index, ] for seq in self.event_seq]
            self.time_delta_seq = [seq + [eos_elapse, ] for seq in self.time_delta_seq]
            self.num_types += 1

    def __len__(self):
        assert len(self.time_seq) == len(self.event_seq) and len(self.time_delta_seq) == len(self.event_seq), \
            f"Inconsistent lengths for data! time_seq_len:{len(self.time_seq)}, event_len: {len(self.event_seq)}, time_delta_seq_len: {len(self.time_delta_seq)}"
        return len(self.event_seq)

    def __getitem__(self, idx):
        return self.time_seq[idx], self.time_delta_seq[idx], self.event_seq[idx]

    def padding(self, seqs, dtype, max_len=None):
        # padding to the max_length
        if max_len is None:
            max_len = max(len(seq) for seq in seqs)
        batch_seq = np.array([seq + [self.pad_index] * (max_len - len(seq)) for seq in seqs], dtype=np.float64)

        # by default, return float32 tensor
        return torch.tensor(batch_seq, dtype=dtype)

    def createConcurrentMask(self, time_seq):
        max_length = max([len(x) for x in time_seq])
        batch_size = len(time_seq)
        mask = torch.ones((batch_size, max_length, max_length), dtype=torch.uint8)
        for _batch_i, _time_seq in enumerate(time_seq):
            # buf[(start, end, can_attend_to_earlier_than)]
            # mask[_batch_i, start:end, 0:can_attend_to_earlier_than] = 0
            if self.add_bos:
                cur_index = 1
                buf = [(0, 1, 1)]
            else:
                cur_index = 0
                buf = []
            cur_time = _time_seq[cur_index]
            last_index = cur_index
            while cur_index < len(_time_seq):
                if abs(_time_seq[cur_index] - cur_time) < self.eps:
                    cur_index += 1
                else:
                    buf.append((last_index, cur_index, last_index))
                    last_index = min(cur_index, len(_time_seq) - 1)
                    cur_time = _time_seq[last_index]

            if buf[-1][1] <= len(_time_seq) - 1:
                buf.append((last_index, cur_index, last_index))

            for item in buf:
                beg, end, attend_to = item
                mask[_batch_i, beg: end, 0: attend_to] = 0
        return mask

    def createPadAttnMask(self, event_seq, concurrent_mask=None):
        # 1 -- pad, 0 -- non-pad
        batch_size, seq_len = event_seq.size(0), event_seq.size(1)
        batch_seq_pad_mask = event_seq.eq(self.pad_index)
        attention_key_pad_mask = batch_seq_pad_mask.unsqueeze(1).expand(batch_size, seq_len, -1)
        subsequent_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=event_seq.device, dtype=torch.uint8), diagonal=0
        ).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = subsequent_mask | attention_key_pad_mask.bool()
        if concurrent_mask is None:
            # no way to judge concurrent events, simply believe there is no concurrent events
            pass
        else:
            attention_mask |= concurrent_mask.bool()
        return ~batch_seq_pad_mask, attention_mask

    def collate_fn(self, batch):
        time_seq, time_delta_seq, event_seq = list(zip(*batch))
        if self.concurrent:
            concurrent_mask = self.createConcurrentMask(time_seq)
        else:
            concurrent_mask = None

        time_seq = self.padding(time_seq, torch.float64)
        time_delta_seq = self.padding(time_delta_seq, torch.float64)
        event_seq = self.padding(event_seq, torch.long)

        batch_non_pad_mask, attention_mask = self.createPadAttnMask(event_seq, concurrent_mask)

        type_mask = torch.zeros([*event_seq.size(), self.event_num])
        for i in range(self.event_num):
            type_mask[:, :, i] = event_seq == i

        return time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask


def create_tpp_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        # num_workers=num_workers,  # comment this line which may cause warnings in ParallelNative.cpp
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle
    )
