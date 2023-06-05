
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class EBMDataset(Dataset):
    def __init__(
            self,
            data,
            num_noise_samples: int,
            time_factor: float = 1.0,
            max_seq_length: int = 10,
            target_name: str = 'relation'
    ):
        super().__init__()
        self.data = data
        self.num_noise_samples = num_noise_samples
        self.time_factor = time_factor
        self.max_seq_length = max_seq_length
        self.target_name = target_name

    def __getitem__(self, index) -> T_co:
        point = self.data[index]
        original_index = point[0]
        label_causal_event_list = point[1]
        noise_causal_events_list = point[2]
        label_target = point[3]
        noise_target = point[4]

        label_dict = list_of_dict_to_dict(label_causal_event_list)
        label_dict = padding_lists_in_dict(label_dict, padding_length=self.max_seq_length)

        noise_dicts = [
            padding_lists_in_dict(
                list_of_dict_to_dict(noise_causal_event_list),
                padding_length=self.max_seq_length
            ) for noise_causal_event_list in noise_causal_events_list
        ]

        assert len(noise_dicts) >= self.num_noise_samples, 'Length not match'

        noise_dict = list_of_dict_to_dict(noise_dicts)

        return {
            'label_seq_subject': label_dict['subject'],
            'label_seq_object': label_dict['object'],
            'label_seq_relation': label_dict['relation'],
            'label_seq_time': label_dict['time'] / self.time_factor,

            'noise_seq_subject': noise_dict['subject'][:self.num_noise_samples],
            'noise_seq_object': noise_dict['object'][:self.num_noise_samples],
            'noise_seq_relation': noise_dict['relation'][:self.num_noise_samples],
            'noise_seq_time': noise_dict['time'][:self.num_noise_samples] / self.time_factor,

            # label information
            'label_target': label_target,  # real rels
            'pred_target': [label_target] + noise_target[:4],  # make it the same length for all events
            'original_index': original_index
        }

    def __len__(self):
        return len(self.data)


class EBMValidDataset(EBMDataset):
    def __init__(
            self,
            data,
            num_noise_samples: int,
            time_factor: float = 1.0,
            max_seq_length: int = 10,
            target_name: str = 'relation'
    ):
        super().__init__(
            data, num_noise_samples, time_factor, max_seq_length, target_name
        )
        self.data = [
            item for item in self.data if len(item[4]) == num_noise_samples
        ]


def list_of_dict_to_dict(list_of_dicts):
    if not list_of_dicts:
        raise ValueError("The list of dicts is empty")

    dict_of_lists = {key: np.array([d[key] for d in list_of_dicts]) for key in list_of_dicts[0]}

    return dict_of_lists


def padding_lists_in_dict(a_dict, padding_length=10, padding_values=None):
    padding_values = padding_values or {}

    for k in a_dict:
        v = a_dict[k]
        len_diff = padding_length - len(v)
        if len_diff < 0:
            a_dict[k] = v[-padding_length:]
        elif len_diff > 0:
            a_dict[k] = np.concatenate([[padding_values.get(k, 0)] * len_diff, v])

    return a_dict
