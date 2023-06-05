
import numpy as np
import yaml
from torch.utils.data import DataLoader

import dictdatabase as DDB

from models.ebm.att_nhp_ebm import AttNHPEBM
from models.ebm.rnn import RNNEbm
from models.ebm_runner import EBMRunner
from preprocess.datasets.ebm_dataset import EBMDataset, EBMValidDataset
from utils.general import setup_seed
from utils.metrics import is_hit, rank
from utils.torch_utils import count_torch_model_params


def main(dataset_name, model_name, pred_type, model_config, num_test_points: int = 15000, is_load: bool = False):
    DDB.config.storage_directory = 'scripts/gdelt/ddb_storage'
    ebm_data = list(DDB.at(f'{model_name}_{dataset_name}_bert_ebm_dataset', pred_type).read().values())

    if pred_type == 'relation':
        num_noise_samples = 4
        top_n = 3
    elif pred_type == 'object':
        num_noise_samples = 19
        top_n = 10
    else:
        num_noise_samples = 99
        top_n = 50

    if dataset_name == 'gdelt':
        num_entities = 2279
        num_relations = 20
    else:
        return

    print('Original length', len(ebm_data))
    train_data = ebm_data[:-num_test_points]
    test_data = ebm_data[-num_test_points:]

    train_dataset = EBMDataset(train_data, num_noise_samples=num_noise_samples, time_factor=100.0, max_seq_length=10)
    # train_dataset = EBMValidDataset(train_data, num_noise_samples=num_noise_samples, time_factor=100.0)  # used when ke_anhp_gdelt_comb
    test_dataset = EBMValidDataset(test_data, num_noise_samples=num_noise_samples, time_factor=100.0, max_seq_length=10)
    print('train', len(train_dataset))
    print('test', len(test_dataset))

    train_dl = DataLoader(
        train_dataset,
        batch_size=model_config['train']['batch_size'],
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=model_config['train']['batch_size'] * 2,
    )
    if model_config['name'].lower() == 'rnn':
        model = RNNEbm(
            num_entities=num_entities,
            num_relations=num_relations,
            num_noise_samples=num_noise_samples,
            embedding_dim=model_config['embedding_dim'],
            num_cells=model_config['num_cells'],
            num_layers=model_config['num_layers'],
            dropout_rate=model_config['dropout_rate']
        )
    else:
        model = AttNHPEBM(
            num_entities=num_entities,
            num_relations=num_relations,
            num_noise_samples=num_noise_samples,
            embedding_dim=model_config['embedding_dim'],
            d_model=model_config['d_model'],
            d_time=model_config['d_time'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            use_ln=False
        )

    runner = EBMRunner(
        model,
        loss_function=model_config['train']['loss_function'],
        lr=float(model_config['train']['lr']),
        log_path=f'logs/{model_name}_{dataset_name}_ebm_{pred_type}.pt',
        lr_scheduler_params=model_config.get('lr_scheduler')
    )

    if is_load:
        runner.load()
    else:
        runner.train(train_dl, valid_dataloader=test_dl, num_epochs=model_config['train']['num_epochs'], verbose=False)
        runner.load()

    _, (label_score, fake_scores) = runner.evaluate_one_epoch(test_dl)
    label = np.zeros_like(label_score, dtype=np.int32)
    pred = np.concatenate([label_score[:, None], fake_scores], axis=-1)

    hit_ratio = np.mean(is_hit(label, pred, top_n=top_n))
    mean_rank = np.mean(rank(label, pred))

    print(
        f'Hit ratio: {hit_ratio}\n'
        f'Mean rank: {mean_rank}'
    )
    return mean_rank


if __name__ == '__main__':
    setup_seed()
    ke_gdelt_ebm_config_fn = 'configs/ke_anhp_gdelt_ebm_rel.yaml'
    with open(ke_gdelt_ebm_config_fn) as config_file:
        config = yaml.safe_load(config_file)

    main(
        dataset_name=config['dataset'],
        model_name=config['base_model'],
        pred_type=config['pred_type'],
        model_config=config['model'],
        is_load=False
    )
