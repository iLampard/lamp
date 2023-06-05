import numpy as np
import torch
from torch.utils.data import Dataset


class TPPEBMDataset(Dataset):
    def __init__(
            self,
            data,
            num_event_types,
            num_noise_samples: int,
            time_factor: float = 1.0,
            max_seq_length: int = 20
    ):
        self.data = data
        self.event_num = num_event_types
        self.pad_index = self.event_num
        self.max_seq_length = max_seq_length
        self.num_noise_samples = num_noise_samples
        self.time_factor = time_factor

        self.real_sample = [data[1] for data in self.data]
        self.fake_sample = [data[2] for data in self.data]

        self.real_time_seqs = []  # [num_seqs, seq_len]
        self.real_type_seqs = []
        self.real_time_delta_seqs = []
        self.fake_time_seqs = []  # [num_seqs, num_fake_sample=4, seq_len]
        self.fake_type_seqs = []
        self.fake_time_delta_seqs = []
        for i in range(len(self.real_sample)):
            real_time_seq = [x['event_time'] for x in self.real_sample[i]]
            real_time_seq = [x - real_time_seq[0] for x in real_time_seq]
            real_time_delta_seq = [0.0] + [y - x for x, y in zip(real_time_seq[:-1], real_time_seq[1:])]
            real_type_seq = [x['event_type'] for x in self.real_sample[i]]
            self.real_time_seqs.append(real_time_seq)
            self.real_type_seqs.append(real_type_seq)
            self.real_time_delta_seqs.append(real_time_delta_seq)

            time_seq = []
            type_seq = []
            time_delta_seq = []

            for j in range(self.num_noise_samples):  # make it the same length
                fake_time_seq = [x['event_time'] for x in self.fake_sample[i][j]]
                fake_time_seq = [x - fake_time_seq[0] for x in fake_time_seq]
                fake_time_delta_seq = [0.0] + [y - x for x, y in zip(fake_time_seq[:-1], fake_time_seq[1:])]
                fake_type_seq = [x['event_type'] for x in self.fake_sample[i][j]]
                time_seq.append(fake_time_seq)
                type_seq.append(fake_type_seq)
                time_delta_seq.append(fake_time_delta_seq)

            self.fake_time_seqs.append(time_seq)
            self.fake_type_seqs.append(type_seq)
            self.fake_time_delta_seqs.append(time_delta_seq)

    def __len__(self):
        """

        Returns: length of the dataset

        """

        return len(self.real_time_seqs)

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            time_seq, time_delta_seq and event_seq element

        """
        real_time_seqs, real_time_delta_seqs, real_type_seqs = self.real_time_seqs[idx], self.real_time_delta_seqs[idx], \
                                                               self.real_type_seqs[idx]

        real_time_seqs, real_time_delta_seqs, real_type_seqs, real_batch_non_pad_mask, real_attention_mask, real_type_mask \
            = self.pad_seqs([real_time_seqs], [real_time_delta_seqs], [real_type_seqs])

        # do a padding here otherwise it will fail in collate_fn
        fake_time_seqs, fake_time_delta_seqs, fake_type_seqs = self.fake_time_seqs[idx], self.fake_time_delta_seqs[idx], \
                                                               self.fake_type_seqs[idx]
        fake_time_seqs, fake_time_delta_seqs, fake_type_seqs, fake_batch_non_pad_mask, fake_attention_mask, fake_type_mask \
            = self.pad_seqs(fake_time_seqs, fake_time_delta_seqs, fake_type_seqs)

        return {'real_time_seqs': real_time_seqs,  # [1, seq_len]
                'real_time_delta_seqs': real_time_delta_seqs,
                'real_type_seqs': real_type_seqs,
                'real_batch_non_pad_mask': real_batch_non_pad_mask,
                'real_attention_mask': real_attention_mask,
                'real_type_mask': real_type_mask,
                'fake_time_seqs': fake_time_seqs,  # [num_sample, seq_len]
                'fake_time_delta_seqs': fake_time_delta_seqs,
                'fake_type_seqs': fake_type_seqs,
                'fake_batch_non_pad_mask': fake_batch_non_pad_mask,
                'fake_attention_mask': fake_attention_mask,
                'fake_type_mask': fake_type_mask
                }

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

    def pad_seqs(self, time_seqs, time_delta_seqs, type_seqs):
        """
        Args:
            batch: batch sequence data

        Returns:
            batch tensors of time_seqs, time_delta_seqs, event_seqs,
            batch_non_pad_mask, attention_mask, type_mask

        """

        time_seqs = self.padding(time_seqs, torch.float32, max_len=self.max_seq_length)
        time_delta_seqs = self.padding(time_delta_seqs, torch.float32, max_len=self.max_seq_length)
        type_seqs = self.padding(type_seqs, torch.long, max_len=self.max_seq_length)

        batch_non_pad_mask, attention_mask = self.createPadAttnMask(type_seqs)

        type_mask = torch.zeros([*type_seqs.size(), self.event_num])
        for i in range(self.event_num):
            type_mask[:, :, i] = type_seqs == i

        return time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask


class TPPEBMValidDataset(TPPEBMDataset):
    def __init__(
            self,
            data,
            num_event_types,
            num_noise_samples: int,
            time_factor: float = 1.0,
            max_seq_length: int = 20
    ):
        super().__init__(
            data, num_event_types, num_noise_samples, time_factor, max_seq_length
        )
        self.data = [
            item for item in self.data if len(item[2]) == num_noise_samples
        ]
        self.num_noise_samples = num_noise_samples
        self.time_factor = time_factor
        self.max_seq_length = max_seq_length
