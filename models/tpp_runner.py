import numpy as np
import torch

from utils.general import get_value_by_key
from utils.metrics import rank, is_hit, time_rmse_np


class TPPRunner():
    def __init__(
            self,
            model,
            lr: float = 1e-3,
            num_epochs: int = 10,
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_dl, valid_dl=None, num_epochs: int = None, verbose: bool = True, autosave: bool = True):
        num_epochs = num_epochs or self.num_epochs

        best_metric = float('inf')
        for epoch_i in range(num_epochs):
            loss = self.train_one_epoch(train_dl, verbose=verbose)
            print('Epoch', epoch_i, 'loss:', loss)
            if valid_dl is not None:
                # metric, _ = self.evaluate_one_epoch(valid_dl)
                metric = self.evaluate_one_epoch_by_loss(valid_dl)
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
        num_batches = len(train_dataloader)
        for i, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            ret_tuple = self.model(
                batch,
                return_loss=True,
            )
            loss = ret_tuple[0]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            loss = loss.detach().cpu().numpy()
            if verbose:
                print(f'--- batch {i} loss:', loss)

            epoch_loss += loss / num_batches

        return epoch_loss

    def evaluate_one_epoch_by_loss(self, dataloader):
        epoch_loss = 0
        num_batches = len(dataloader)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                if batch[0].size(1) <= 2:
                    continue

                ret_tuple = self.model(
                    batch,
                    return_loss=True,
                )
                loss = ret_tuple[0]

                loss = loss.detach().cpu().numpy()

                epoch_loss += loss / num_batches

        return epoch_loss

    def evaluate_one_epoch(self, dataloader, top_n_hit: int = 5):
        self.model.eval()

        metric_list_dict = {
            'type': {
                'mean_rank': [],
                'hit_ratio': []
            },
            'time': {
                'rmse': []
            },
        }

        res_batches = []
        label_dtime_batch = []
        pred_dtime_batch = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch_pred = self.model.inference(
                    batch
                )

                if len(batch_pred):
                    label_type = np.array(get_value_by_key('label_type', batch_pred))
                    pred_type_score = np.array(get_value_by_key('pred_type_score', batch_pred))
                    label_dtime = np.array(get_value_by_key('label_dtime', batch_pred))
                    pred_dtime = np.array(get_value_by_key('pred_dtime', batch_pred))

                    mean_rank = rank(label=label_type, pred=pred_type_score).tolist()
                    metric_list_dict['type']['mean_rank'].extend(mean_rank)

                    hit_ratio = is_hit(label=label_type, pred=pred_type_score,
                                       top_n=top_n_hit)

                    metric_list_dict['type']['hit_ratio'].extend(hit_ratio)
                    label_dtime_batch.extend(label_dtime)
                    pred_dtime_batch.extend(pred_dtime)

                    res_batches.append(batch_pred)

        for pred_type, metric_dict in metric_list_dict.items():
            for metric_name, metric_list in metric_dict.items():
                if len(metric_list) > 0:
                    # set the important metric
                    if pred_type == 'type' and metric_name == 'mean_rank':
                        metric = np.mean(metric_list)
                    print(f'--------- {pred_type}-{metric_name}:', np.mean(metric_list))

        # print(f'--------- RMSE:', time_rmse_np(pred_dtime_batch, label_dtime_batch))

        return metric, res_batches

    def save(self, path: str = 'anhp.pt'):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = 'anhp.pt'):
        self.model.load_state_dict(torch.load(path))

    def is_empty_pred(self, pred_output):
        if len(pred_output) == 0:
            return True
        elif len(pred_output[0]) == 0:
            return True
        return False
