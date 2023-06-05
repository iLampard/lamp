
import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import is_hit, rank


class KGRunner():
    def __init__(
            self,
            model,
            source_data,
            lr: float = 1e-3,
            num_epochs: int = 10,
            storage_uri: str = None,
            device: str = 'cpu'
    ):
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.storage_uri = storage_uri or 'ke.pt'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.source_data = source_data

        # default parameters
        self.clipping_value = 5.0

        # # clip gradient
        # for p in self.model.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clipping_value, clipping_value))

    def train(self, train_dl, valid_dl=None, num_epochs: int = None, verbose: bool = True, autosave: bool = True):
        num_epochs = num_epochs or self.num_epochs

        best_metric = float('inf')
        for epoch_i in range(num_epochs):
            loss = self.train_one_epoch(train_dl, verbose=verbose)
            print('Epoch', epoch_i, 'loss:', loss)
            if valid_dl is not None:
                metric, _ = self.evaluate_one_epoch(valid_dl)
                if metric < best_metric:
                    if autosave:
                        self.save()
                        print('save model at epoch', epoch_i)
                    best_metric = metric
                    print('------------ Best metric:', metric)

    def train_one_epoch(
            self,
            train_dataloader,
            verbose: bool = True
    ):
        epoch_loss = 0
        self.model.zero_states()
        num_batches = len(train_dataloader)
        for i, batch in tqdm(enumerate(train_dataloader)):
            self.optimizer.zero_grad()
            ret_tuple = self.model(
                seq_subject=batch['seq_subject'].to(self.device),
                seq_object=batch['seq_object'].to(self.device),
                seq_time=batch['seq_time'].to(self.device),
                seq_relation=batch['seq_relation'].to(self.device),
                return_loss=True,
            )
            loss = ret_tuple[0]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clipping_value)
            self.optimizer.step()

            loss = loss.detach().cpu().numpy()
            if verbose:
                print(f'--- batch {i} loss:', loss)

            epoch_loss += loss / num_batches

        return epoch_loss

    def evaluate_one_epoch(
            self,
            dataloader,
            warmup_steps: int = 0,
            with_index: bool = False,
            predict_relation: bool = True,
            predict_object: bool = False,
            top_n_hit: int = 5
    ):
        self.model.eval()

        metric_list_dict = {
            'relation': {
                'mean_rank': [],
                'hit_ratio': []
            },
            'object': {
                'mean_rank': [],
                'hit_ratio': []
            },
        }
        res_batches = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                if with_index:
                    original_idx = batch[0]
                    batch = batch[1]

                if i >= warmup_steps:
                    # relation prediction
                    batch_pred = self.model.inference(
                        seq_subject=batch['seq_subject'].to(self.device),
                        seq_object=batch['seq_object'].to(self.device),
                        seq_time=batch['seq_time'].to(self.device),
                        seq_relation=batch['seq_relation'].to(self.device),
                        predict_relation=predict_relation,
                        predict_object=predict_object,
                    )

                    if predict_relation:
                        batch_pred['relation'] = batch['seq_relation'][:, -1].detach().cpu().numpy()

                        # calculate metric
                        mean_rank = rank(label=batch_pred['relation'],
                                         pred=batch_pred['pred_relation']).tolist()
                        hit_ratio = is_hit(label=batch_pred['relation'],
                                           pred=batch_pred['pred_relation'],
                                           top_n=top_n_hit)

                        metric_list_dict['relation']['mean_rank'].extend(mean_rank)
                        metric_list_dict['relation']['hit_ratio'].extend(hit_ratio)

                    if predict_object:
                        batch_pred['object'] = batch['seq_object'][:, -1].detach().cpu().numpy()

                        # calculate metric
                        mean_rank = rank(label=batch_pred['object'],
                                         pred=batch_pred['pred_object']).tolist()
                        hit_ratio = is_hit(label=batch_pred['object'], pred=batch_pred['pred_object'],
                                           top_n=top_n_hit)

                        metric_list_dict['object']['mean_rank'].extend(mean_rank)
                        metric_list_dict['object']['hit_ratio'].extend(hit_ratio)

                    res_batches.append(batch_pred)

                    if with_index:
                        batch_pred['original_idx'] = original_idx

                # forward
                self.model(
                    seq_subject=batch['seq_subject'].to(self.device),
                    seq_object=batch['seq_object'].to(self.device),
                    seq_time=batch['seq_time'].to(self.device),
                    seq_relation=batch['seq_relation'].to(self.device),
                    return_loss=False,
                    is_update_last=True
                )
        metric = None
        for pred_type, metric_dict in metric_list_dict.items():
            for metric_name, metric_list in metric_dict.items():
                if len(metric_list) > 0:
                    # set the important metric
                    if pred_type == 'relation' and metric_name == 'mean_rank':
                        metric = np.mean(metric_list)
                    if metric_name == 'hit_ratio':
                        metric_name = metric_name + str(top_n_hit)
                    print(f'--------- {pred_type}-{metric_name}:', np.mean(metric_list))

        return metric, res_batches

    def evaluate_both_one_epoch(
            self,
            dataloader,
            warmup_steps: int = 0,
            with_index: bool = False,
            top_n_hit: int = 5
    ):
        self.model.eval()

        metric_list_dict = {
            'relation': {
                'mean_rank': [],
                'hit_ratio': []
            },
            'object': {
                'mean_rank': [],
                'hit_ratio': []
            },
        }
        res_batches = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                if with_index:
                    original_idx = batch[0]
                    batch = batch[1]

                if i >= warmup_steps:
                    # relation prediction
                    batch_pred = self.model.inference_both(
                        seq_subject=batch['seq_subject'].to(self.device),
                        seq_object_minus=batch['seq_object'][..., :-1].to(self.device),
                        seq_time=batch['seq_time'].to(self.device),
                        seq_relation_minus=batch['seq_relation'][..., :-1].to(self.device),
                    )

                    batch_pred['relation'] = batch['seq_relation'][:, -1].detach().cpu().numpy()

                    # calculate metric
                    mean_rank = rank(label=batch_pred['relation'],
                                     pred=batch_pred['pred_relation']).tolist()
                    hit_ratio = is_hit(label=batch_pred['relation'],
                                       pred=batch_pred['pred_relation'],
                                       top_n=top_n_hit)

                    metric_list_dict['relation']['mean_rank'].extend(mean_rank)
                    metric_list_dict['relation']['hit_ratio'].extend(hit_ratio)

                    batch_pred['object'] = batch['seq_object'][:, -1].detach().cpu().numpy()

                    # calculate metric
                    mean_rank = rank(label=batch_pred['object'],
                                     pred=batch_pred['pred_object']).tolist()
                    hit_ratio = is_hit(label=batch_pred['object'], pred=batch_pred['pred_object'],
                                       top_n=top_n_hit)

                    metric_list_dict['object']['mean_rank'].extend(mean_rank)
                    metric_list_dict['object']['hit_ratio'].extend(hit_ratio)

                    res_batches.append(batch_pred)

                    if with_index:
                        batch_pred['original_idx'] = original_idx

                # forward
                self.model(
                    seq_subject=batch['seq_subject'].to(self.device),
                    seq_object=batch['seq_object'].to(self.device),
                    seq_time=batch['seq_time'].to(self.device),
                    seq_relation=batch['seq_relation'].to(self.device),
                    return_loss=False,
                    is_update_last=True
                )
        metric = None
        for pred_type, metric_dict in metric_list_dict.items():
            for metric_name, metric_list in metric_dict.items():
                if len(metric_list) > 0:
                    # set the important metric
                    if pred_type == 'relation' and metric_name == 'mean_rank':
                        metric = np.mean(metric_list)
                    if metric_name == 'hit_ratio':
                        metric_name = metric_name + str(top_n_hit)
                    print(f'--------- {pred_type}-{metric_name}:', np.mean(metric_list))

        return metric, res_batches

    def evaluate_combination_one_epoch(
            self,
            dataloader,
            warmup_steps: int = 0,
            with_index: bool = False,
            rel_topk: int = 5,
            obj_topk: int = 20
    ):
        """

        Remark: batch_size is 1.

        Args:
            dataloader:
            warmup_steps:
            with_index:

        Returns:

        """
        self.model.eval()

        metric_list_dict = {
            'relation': {
                'mean_rank': [],
                'hit_ratio': []
            },
            'object': {
                'mean_rank': [],
                'hit_ratio': []
            },
        }
        res_batches = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                if with_index:
                    original_idx = batch[0]
                    batch = batch[1]

                if i >= warmup_steps:

                    # relation prediction
                    batch_pred = self.model.inference_combination(
                        seq_subject=batch['seq_subject'].to(self.device),
                        seq_object_minus=batch['seq_object'][..., :-1].to(self.device),
                        seq_time=batch['seq_time'].to(self.device),
                        seq_relation_minus=batch['seq_relation'][..., :-1].to(self.device),
                        rel_topk=rel_topk,
                        obj_topk=obj_topk,
                    )

                    batch_pred['relation'] = batch['seq_relation'][:, -1].detach().cpu().numpy()
                    batch_pred['object'] = batch['seq_object'][:, -1].detach().cpu().numpy()

                    # calculate metric
                    mean_rank = rank(label=batch_pred['relation'],
                                     pred=batch_pred['pred_relation']).tolist()
                    hit_ratio = is_hit(label=batch_pred['relation'],
                                       pred=batch_pred['pred_relation'],
                                       top_n=rel_topk)

                    metric_list_dict['relation']['mean_rank'].extend(mean_rank)
                    metric_list_dict['relation']['hit_ratio'].extend(hit_ratio)

                    # calculate metric
                    mean_rank = rank(label=batch_pred['object'],
                                     pred=batch_pred['pred_object']).tolist()
                    hit_ratio = is_hit(label=batch_pred['object'], pred=batch_pred['pred_object'],
                                       top_n=obj_topk)

                    metric_list_dict['object']['mean_rank'].extend(mean_rank)
                    metric_list_dict['object']['hit_ratio'].extend(hit_ratio)

                    res_batches.append(batch_pred)

                    if with_index:
                        batch_pred['original_idx'] = original_idx

                # forward
                self.model(
                    seq_subject=batch['seq_subject'].to(self.device),
                    seq_object=batch['seq_object'].to(self.device),
                    seq_time=batch['seq_time'].to(self.device),
                    seq_relation=batch['seq_relation'].to(self.device),
                    return_loss=False,
                    is_update_last=True
                )
        metric = None
        for pred_type, metric_dict in metric_list_dict.items():
            for metric_name, metric_list in metric_dict.items():
                if len(metric_list) > 0:
                    # set the important metric
                    if pred_type == 'relation' and metric_name == 'mean_rank':
                        metric = np.mean(metric_list)
                    if pred_type == 'relation' and metric_name == 'hit_ratio':
                        metric_name = metric_name + str(rel_topk)
                    if pred_type == 'object' and metric_name == 'hit_ratio':
                        metric_name = metric_name + str(obj_topk)
                    print(f'--------- {pred_type}-{metric_name}:', np.mean(metric_list))

        return metric, res_batches

    def save(self, path: str = None):
        path = path or self.storage_uri
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = None):
        path = path or self.storage_uri
        self.model.load_state_dict(torch.load(path, map_location=self.device))
