import yaml
from torch.utils.data import DataLoader

from preprocess.datasets.tpp_dataset import TPPDataset
from utils.general import file_uri_reader_processor


class TPPDatasetFactory:
    def __init__(
            self,
            data,
            num_event_types: int,
            train_end_date: str = '2015-08-01',
            valid_end_date: str = '2016-02-01',
            time_factor: float = 1.0,
    ):
        # data is the list of tuple
        self.data = data
        self.num_event_types = num_event_types
        self.train_end_date = str(train_end_date)
        self.valid_end_date = str(valid_end_date)
        self.time_factor = time_factor

        # set datasets
        self.train_dataset = TPPDataset(
            data=self.data,
            time_factor=self.time_factor,
            end_date=self.train_end_date,
            num_event_types=num_event_types
        )

        if self.valid_end_date is None:
            self.valid_dataset = None
        else:
            self.valid_dataset = TPPDataset(
                data=self.data,
                time_factor=self.time_factor,
                start_date=self.train_end_date,
                end_date=self.valid_end_date,
                num_event_types=num_event_types
            )

        if self.valid_end_date is None:
            self.test_dataset = None
        else:
            self.test_dataset = TPPDataset(
                data=self.data,
                time_factor=self.time_factor,
                start_date=self.valid_end_date,
                num_event_types=num_event_types
            )

    def get_train_dataloader(self, **kwargs):
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 1
        return DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            **kwargs
        )

    def get_valid_dataloader(self, **kwargs):
        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            collate_fn=self.valid_dataset.collate_fn,
            **kwargs
        )

    def get_test_dataloader(self, **kwargs):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate_fn,
            **kwargs
        )

    @staticmethod
    def build_from_config_dict(config_dict: dict):
        data_obj = file_uri_reader_processor(config_dict['data_dir'])

        return TPPDatasetFactory(
            data=data_obj['user_seqs'],
            num_event_types=config_dict.get('num_event_types', data_obj.get('dim_process')),
            train_end_date=config_dict['train_end_date'],
            valid_end_date=config_dict['valid_end_date'],
            time_factor=config_dict.get('time_factor', 1.0),
        )

    @staticmethod
    def build_from_config_file(config_file: str):
        with open(config_file) as config_file:
            config = yaml.safe_load(config_file)
        return TPPDatasetFactory.build_from_config_dict(config['data'])
