import dictdatabase as DDB
import numpy as np
import yaml
from torch.utils.data import DataLoader

from models.ebm.att_nhp_ebm_amazon import AttNHPEBMTPP
from models.ebm_runner import EBMRunner
from preprocess.datasets.tpp_emb_dataset import TPPEBMDataset, TPPEBMValidDataset
from utils.general import setup_seed
from utils.metrics import is_hit, rank


def main(dataset_name, model_name, pred_type, model_config, num_test_points: int = 400, is_load: bool = False):
    DDB.config.storage_directory = 'scripts/amazon/ddb_storage'
    ebm_data = list(DDB.at(f'{model_name}_{dataset_name}_ebm_dataset_time_bak', pred_type).read().values())

    model = AttNHPEBMTPP(model_config)

    if pred_type == 'type':
        num_noise_samples = 4
        top_n = 3
        num_event_types = 24
    elif pred_type == 'dtime':
        top_n = 5
        num_event_types = 24
        num_noise_samples = 3
    else:
        raise RuntimeError(f'Unknown pred_type {pred_type}')

    print('Original length', len(ebm_data))
    train_data = ebm_data[:-num_test_points]
    test_data = ebm_data[-num_test_points:]

    train_dataset = TPPEBMDataset(train_data, num_event_types=num_event_types, num_noise_samples=num_noise_samples)
    test_dataset = TPPEBMDataset(test_data, num_event_types=num_event_types, num_noise_samples=num_noise_samples)
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

    model_log_path = f'logs/{model_name}_{dataset_name}_ebm_{pred_type}.pt'
    runner = EBMRunner(
        model,
        is_tpp_model=True,
        loss_function=model_config['train']['loss_function'],
        lr=float(model_config['train']['lr']),
        log_path=model_log_path,
        lr_scheduler_params=model_config['train'].get('lr_scheduler')
    )

    if is_load:
        runner.load()
    else:
        runner.train(train_dl, valid_dataloader=test_dl, num_epochs=model_config['train']['num_epochs'], verbose=False)
        runner.load()

    if pred_type == 'type':
        _, (label_score, fake_scores) = runner.evaluate_one_epoch(test_dl)
        label = np.zeros_like(label_score, dtype=np.int32)
        pred = np.concatenate([label_score[:, None], fake_scores], axis=-1)

        hit_ratio = np.mean(is_hit(label, pred, top_n=top_n))
        metric = np.mean(rank(label, pred))

        print(
            f'Hit ratio: {hit_ratio}\n'
            f'Mean rank: {metric}'
        )
    else:
        metric, _ = runner.evaluate_one_epoch_time(test_dl)

    return metric


if __name__ == '__main__':
    setup_seed()
    amazon_ebm_config_fn = 'configs/amazon_anhp_ebm_type.yaml'
    with open(amazon_ebm_config_fn) as config_file:
        config = yaml.safe_load(config_file)

    main(
        dataset_name=config['dataset'],
        model_name=config['base_model'],
        pred_type=config['pred_type'],
        model_config=config['model'],
        is_load=False
    )
