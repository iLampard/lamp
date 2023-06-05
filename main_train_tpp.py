
import yaml

from models.know_evolve import KnowEvolve
from models.tpp.att_nhp import AttNHP
from models.kg_runner import KGRunner
from models.tpp_runner import TPPRunner
from preprocess.kg_data_factory import KGDatasetFactory
from preprocess.tpp_data_factory import TPPDatasetFactory
from utils.general import file_uri_writer_processor


def build_data_factory_and_runner_from_config_file(config_name):
    with open(config_name) as config_file:
        config = yaml.safe_load(config_file)
    data_config = config['data']
    model_config = config['model']

    if data_config['type'].lower() == 'kg':
        dataset_factory = KGDatasetFactory.build_from_config_dict(data_config)
    else:
        dataset_factory = TPPDatasetFactory.build_from_config_dict(data_config)

    if model_config['name'].lower() == 'ke':
        model = KnowEvolve(
            num_entities=dataset_factory.num_entities,
            num_relations=dataset_factory.num_relations,
            dim_c=model_config['dim_c'],
            dim_l=model_config['dim_l'],
            dim_d=model_config['dim_d'],
        )
        runner = KGRunner(
            model,
            source_data=dataset_factory.data,
            lr=model_config.get('lr', 0.001),
            num_epochs=model_config.get('num_epochs', 10),
        )
    elif model_config['name'].lower() == 'ke-tpp':
        pass
    else:
        model = AttNHP(model_config)
        runner = TPPRunner(model,
                           lr=model_config.get('lr', 0.001),
                           num_epochs=model_config.get('num_epochs', 10),
                           )

    return dataset_factory, runner, model_config


if __name__ == '__main__':
    # config_filename = 'configs/ke.yaml'
    config_filename = 'configs/amazon_anhp.yaml'  # tpp

    dataset_factory, runner, model_config = build_data_factory_and_runner_from_config_file(config_filename)

    runner.train(
        train_dl=dataset_factory.get_train_dataloader(batch_size=model_config['train']['batch_size']),
        valid_dl=dataset_factory.get_valid_dataloader(),
        num_epochs=model_config['train']['num_epoch']
    )
    runner.save()

    metric, res = runner.evaluate_one_epoch(
        dataset_factory.get_train_dataloader()
    )

    file_uri_writer_processor(res, 'tpp_amazon_test.pkl')
