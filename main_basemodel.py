import yaml

from models.ke_anhp import KnowEvolveANHP
from models.kg_runner import KGRunner
from models.tpp.att_nhp import AttNHP
from models.tpp_runner import TPPRunner
from preprocess.kg_data_factory import KGDatasetFactory
from preprocess.tpp_data_factory import TPPDatasetFactory
from utils.general import file_uri_writer_processor, setup_seed

# device = 'cuda:0'
from utils.torch_utils import count_torch_model_params

device = 'cpu'


def build_data_factory_and_runner_from_config_file(config_name):
    with open(config_name) as config_file:
        config = yaml.safe_load(config_file)
    data_config = config['data']
    model_config = config['model']

    if data_config['type'].lower() == 'kg':
        dataset_factory = KGDatasetFactory.build_from_config_dict(data_config)
    else:
        dataset_factory = TPPDatasetFactory.build_from_config_dict(data_config)

    if model_config['name'].lower() == 'ke_anhp':
        model = KnowEvolveANHP(
            num_entities=dataset_factory.num_entities,
            num_relations=dataset_factory.num_relations,
            dim_c=model_config['dim_c'],
            dim_l=model_config['dim_l'],
            dim_d=model_config['dim_d'],
            num_layers=model_config['num_layers'],
            n_heads=model_config['n_heads'],
            dropout_rate=model_config['dropout_rate'],
        )
        runner = KGRunner(
            model,
            source_data=dataset_factory.data,
            lr=model_config.get('lr', 0.001),
            num_epochs=model_config.get('num_epochs', 10),
            storage_uri=model_config.get('storage_uri'),
            device=device
        )
    else:
        model = AttNHP(model_config)
        runner = TPPRunner(model,
                           lr=model_config.get('lr', 0.001),
                           num_epochs=model_config.get('num_epochs', 10),
                           )

    return dataset_factory, runner


if __name__ == '__main__':
    setup_seed()
    ke_anhp_config_fn = 'configs/ke_anhp_gdelt.yaml'

    dataset_factory, runner = build_data_factory_and_runner_from_config_file(ke_anhp_config_fn)

    runner.train(
        train_dl=dataset_factory.get_train_dataloader(),
        valid_dl=dataset_factory.get_valid_dataloader(),
        verbose=False
    )

    metric, res = runner.evaluate_one_epoch(
        dataset_factory.iterate_dataset_with_original_index(dataset_factory.test_dataset),
        with_index=True,
        warmup_steps=5000,
        predict_relation=True,
        predict_object=True,
    )
    metric, res = runner.evaluate_combination_one_epoch(
        dataset_factory.iterate_dataset_with_original_index(dataset_factory.test_dataset),
        with_index=True,
        warmup_steps=5000,
    )
    file_uri_writer_processor(res, 'logs/ke_anhp_gdelt_test.pkl')
