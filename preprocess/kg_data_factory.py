
import yaml
from torch.utils.data import Sampler, DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate

from preprocess.datasets.kg_dataset import KGDataset
from utils.general import file_uri_reader_processor


class KGDatasetFactory:
    def __init__(
            self,
            data,
            context_length: int,
            num_entities: int,
            num_relations: int,
            train_end_index_ratio: float = 0.7,
            valid_end_index_ratio: float = 0.8,
            test_end_index_ratio: float = 1.0,
            time_factor: float = 1.0,
            batch_size: int = 32,
    ):
        # data is the list of tuple
        self.data = data
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.context_length = context_length
        self.train_end_index_ratio = train_end_index_ratio
        self.valid_end_index_ratio = valid_end_index_ratio
        self.test_end_index_ratio = test_end_index_ratio or 1.0
        self.time_factor = time_factor
        self.batch_size = batch_size

        # set datasets
        self.train_dataset = KGDataset(
            data=self.data,
            context_length=self.context_length,
            time_factor=self.time_factor,
            end_ratio=self.train_end_index_ratio,
            overlap=False
        )

        if self.valid_end_index_ratio is None or self.train_end_index_ratio >= self.valid_end_index_ratio:
            self.valid_dataset = None
        else:
            self.valid_dataset = KGDataset(
                data=self.data,
                context_length=self.context_length,
                time_factor=self.time_factor,
                start_ratio=self.train_end_index_ratio,
                end_ratio=self.valid_end_index_ratio,
                overlap=True
            )

        if self.valid_end_index_ratio is None:
            start_ratio = self.train_end_index_ratio
        else:
            start_ratio = self.valid_end_index_ratio
        if start_ratio >= self.test_end_index_ratio:
            self.test_dataset = None
        else:
            self.test_dataset = KGDataset(
                data=self.data,
                context_length=self.context_length,
                time_factor=self.time_factor,
                start_ratio=start_ratio,
                end_ratio=self.test_end_index_ratio,
                overlap=True
            )

    def get_train_dataloader(self, **kwargs):
        return DataLoader(
            self.train_dataset,
            batch_sampler=KGTrainingBatchSampler(SequentialSampler(self.train_dataset), self.train_dataset),
            **kwargs
        )

    def get_valid_dataloader(self, **kwargs):
        if kwargs.get('batch_size'):
            del kwargs['batch_size']
        return DataLoader(
            self.valid_dataset,
            # batch_size=self.batch_size,
            batch_size=1,
            **kwargs
        )

    def get_test_dataloader(self, **kwargs):
        return DataLoader(
            self.test_dataset,
            # batch_size=self.batch_size,
            batch_size=1,
            **kwargs
        )

    def iterate_dataset_with_original_index(self, dataset: KGDataset):
        for idx in range(len(dataset)):
            original_idx = dataset.get_original_index(idx)
            batch = default_collate([dataset[idx]])
            yield original_idx, batch

    @staticmethod
    def build_from_config_dict(config_dict: dict):
        data_obj = file_uri_reader_processor(config_dict['data_dir'])

        return KGDatasetFactory(
            data=data_obj['data'],
            context_length=config_dict['context_length'],
            num_relations=config_dict.get('num_relations', data_obj.get('num_rel')),
            num_entities=config_dict.get('num_entities', data_obj.get('num_entity')),
            train_end_index_ratio=config_dict['train_end_index_ratio'],
            valid_end_index_ratio=config_dict['valid_end_index_ratio'],
            test_end_index_ratio=config_dict['test_end_index_ratio'],
            time_factor=config_dict.get('time_factor', 1.0),
            batch_size=config_dict.get('batch_size', 32),
        )

    @staticmethod
    def build_from_config_file(config_file: str):
        with open(config_file) as config_file:
            config = yaml.safe_load(config_file)
        return KGDatasetFactory.build_from_config_dict(config['data'])


class KGTrainingBatchSampler(Sampler):

    def __init__(self, sampler: Sampler[int], kg_dataset: KGDataset) -> None:
        super(KGTrainingBatchSampler, self).__init__(kg_dataset)

        self.sampler = sampler
        self.kg_dataset = kg_dataset

        self.batches = list(self._batches_iterator())

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def _batches_iterator(self):
        batch = []
        existed_entities = set()
        for idx in self.sampler:

            sample = self.kg_dataset[idx]
            sample_entities = set(sample['seq_subject']).union(set(sample['seq_object']))
            if len(existed_entities) == 0 or len(existed_entities.intersection(sample_entities)) == 0:
                # no intersection or first sample
                batch.append(idx)
                existed_entities = existed_entities.union(sample_entities)
            else:
                yield batch
                batch = [idx]
                existed_entities = sample_entities
        yield batch
