
import numpy as np
import torch


class EBMRunner:
    def __init__(
            self,
            model,
            is_tpp_model: bool = False,
            lr: float = 1e-3,
            loss_function: str = 'bce',
            log_path=None,
            lr_scheduler_params=None,
    ):
        self.model = model
        self.is_tpp_model = is_tpp_model
        self.loss_function_name = loss_function
        self.log_path = log_path
        self.loss_function = FunctionRepository.by_name(loss_function)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        lr_scheduler_params = lr_scheduler_params or {}
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=lr_scheduler_params.get('step_size', 10),
            gamma=lr_scheduler_params.get('gamma', 1.0),
            verbose=True
        )

    def train(self, train_dataloader, valid_dataloader=None, num_epochs: int = 5, verbose: bool = True):
        best_metric = float('inf')
        for epoch_i in range(num_epochs):
            loss = self.train_one_epoch(train_dataloader, verbose=verbose)
            self.lr_scheduler.step()
            print('Epoch', epoch_i, 'loss:', loss)
            if valid_dataloader is not None:
                metric, _ = self.evaluate_one_epoch(valid_dataloader)
                # metric, _ = self.evaluate_one_epoch_time(valid_dataloader)
                if metric < best_metric:
                    best_metric = metric
                    self.save()
                    print('------------ Best metric:', metric)

    def train_one_epoch(
            self,
            train_dataloader,
            verbose: bool = True
    ):
        epoch_loss = 0
        num_batches = len(train_dataloader)
        self.model.train()
        for i, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            # # shape -> [N]
            # real_energy_unnorm = self.model(
            #     noise_seq_subject=batch['label_seq_subject'],
            #     noise_seq_object=batch['label_seq_object'],
            #     noise_seq_relation=batch['label_seq_relation'],
            #     noise_seq_time=batch['label_seq_time'],
            # )
            #
            # # shape -> [N, num_samples]
            # fake_energys_unnorm = self.model(
            #     noise_seq_subject=batch['noise_seq_subject'],
            #     noise_seq_object=batch['noise_seq_object'],
            #     noise_seq_relation=batch['noise_seq_relation'],
            #     noise_seq_time=batch['noise_seq_time'],
            # )
            out_energys = self.model(**self._concat_real_and_fakes(batch))
            real_energy_unnorm = out_energys[:, 0]
            fake_energys_unnorm = out_energys[:, 1:]

            loss = self.loss_function(real_energy_unnorm, fake_energys_unnorm)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            loss = loss.detach().cpu().numpy()
            if verbose:
                print(f'--- batch {i} loss:', loss)

            epoch_loss += loss / num_batches

        return epoch_loss

    def evaluate_one_epoch(self, dataloader):
        self.model.eval()

        label_score_list = []
        fake_scores_list = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # # shape -> [N]
                # real_energy_unnorm = self.model(
                #     noise_seq_subject=batch['label_seq_subject'],
                #     noise_seq_object=batch['label_seq_object'],
                #     noise_seq_relation=batch['label_seq_relation'],
                #     noise_seq_time=batch['label_seq_time'],
                # )
                #
                # # shape -> [N, num_samples]
                # fake_energys_unnorm = self.model(
                #     noise_seq_subject=batch['noise_seq_subject'],
                #     noise_seq_object=batch['noise_seq_object'],
                #     noise_seq_relation=batch['noise_seq_relation'],
                #     noise_seq_time=batch['noise_seq_time'],
                # )
                out_energys = self.model(**self._concat_real_and_fakes(batch))
                real_energy_unnorm = out_energys[:, 0]
                fake_energys_unnorm = out_energys[:, 1:]

                if self.loss_function_name == 'bce':
                    real_score, fake_scores = bce_score(real_energy_unnorm, fake_energys_unnorm)
                else:
                    real_score, fake_scores = mnce_score(real_energy_unnorm, fake_energys_unnorm)

                label_score_list.append(real_score)
                fake_scores_list.append(fake_scores)

        # [B]
        label_score_list = torch.cat(label_score_list, dim=0).detach().cpu().numpy()
        # [B, num_samples]
        fake_scores_list = torch.cat(fake_scores_list, dim=0).detach().cpu().numpy()

        mean_rank = (fake_scores_list >= label_score_list[:, None]).sum(axis=-1).mean() + 1
        print('------ Mean rank:', mean_rank)
        print('------ Mean real score:', label_score_list.mean())
        print('------ Mean fake score:', fake_scores_list.mean())

        return mean_rank, (label_score_list, fake_scores_list)

    def evaluate_one_epoch_time(self, dataloader):
        self.model.eval()

        time_error_list = []
        all_cand_time_error_list = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                out_energys = self.model(**self._concat_real_and_fakes(batch))
                real_energy_unnorm = out_energys[:, 0]
                fake_energys_unnorm = out_energys[:, 1:]

                if self.loss_function_name == 'bce':
                    real_score, fake_scores = bce_score(real_energy_unnorm, fake_energys_unnorm)
                else:
                    real_score, fake_scores = mnce_score(real_energy_unnorm, fake_energys_unnorm)

                # shape -> [B]
                best_fake_indices = torch.argmax(fake_scores, dim=-1)
                for j in range(fake_scores.shape[0]):
                    real_time_seq = batch['real_time_seqs'][j, 0, :]
                    real_time_seq_mask = batch['real_batch_non_pad_mask'][j, 0, :]
                    label_time = real_time_seq[real_time_seq_mask][-1]

                    best_fake_time_seq = batch['fake_time_seqs'][j, best_fake_indices[j]]
                    best_fake_time_seq_mask = batch['fake_batch_non_pad_mask'][j, best_fake_indices[j]]
                    best_fake_time = best_fake_time_seq[best_fake_time_seq_mask][-1]
                    error = (best_fake_time - label_time).detach().cpu().numpy()
                    time_error_list.append(error)

                    for fake_card_time_seq, fake_card_time_seq_mask in zip(torch.unbind(batch['fake_time_seqs'][j], dim=0), torch.unbind(batch['fake_batch_non_pad_mask'][j], dim=0)):
                        tmp_time_error = (fake_card_time_seq[fake_card_time_seq_mask][-1] - label_time).detach().cpu().numpy()
                        all_cand_time_error_list.append(tmp_time_error)

        rmse = np.square(time_error_list).mean() ** 0.5
        all_cand_rmse = np.square(all_cand_time_error_list).mean() ** 0.5
        print('------ time RMSE:', rmse)
        print('------ all candidates time RMSE:', all_cand_rmse)

        return rmse, (None, None)

    def save(self, path: str = None):
        if path is None:
            path = self.log_path
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = None):
        if path is None:
            path = self.log_path
        self.model.load_state_dict(torch.load(path))

    def _concat_real_and_fakes(self, batch):
        if not self.is_tpp_model:
            out = {
                'noise_seq_subject': torch.cat([batch['label_seq_subject'][:, None, :], batch['noise_seq_subject']],
                                               dim=-2),
                'noise_seq_object': torch.cat([batch['label_seq_object'][:, None, :], batch['noise_seq_object']],
                                              dim=-2),
                'noise_seq_relation': torch.cat([batch['label_seq_relation'][:, None, :], batch['noise_seq_relation']],
                                                dim=-2),
                'noise_seq_time': torch.cat([batch['label_seq_time'][:, None, :], batch['noise_seq_time']], dim=-2),
            }
        else:
            out = {
                'time_seqs': torch.cat([batch['real_time_seqs'], batch['fake_time_seqs']],
                                       dim=-2),  # [batch_size, 1+num_fake_samples, seq_len]
                'time_delta_seqs': torch.cat([batch['real_time_delta_seqs'], batch['fake_time_delta_seqs']],
                                             dim=-2),  # [batch_size, 1+num_fake_samples, seq_len]
                'type_seqs': torch.cat([batch['real_type_seqs'], batch['fake_type_seqs']],
                                       dim=-2),  # [batch_size, 1+num_fake_samples, seq_len]
                'batch_non_pad_mask': torch.cat([batch['real_batch_non_pad_mask'], batch['fake_batch_non_pad_mask']],
                                                dim=-2),  # [batch_size, 1+num_fake_samples, seq_len]
                'attention_mask': torch.cat([batch['real_attention_mask'], batch['fake_attention_mask']],
                                            dim=-3),  # [batch_size, 1+num_fake_samples, seq_len, seq_len]
                'type_mask': torch.cat([batch['real_type_mask'], batch['fake_type_mask']],
                                       dim=-3),  # [batch_size, 1+num_fake_samples, seq_len, num_event_types]
            }
        return out


class FunctionRepository:
    functions = {}

    @staticmethod
    def register(name):
        def decorator(func):
            FunctionRepository.functions[name] = func
            return func

        return decorator

    @staticmethod
    def by_name(name):
        return FunctionRepository.functions.get(name)


@FunctionRepository.register(name='bce')
def bce_loss(real_energy_unnorm, fake_energys_unnorm):
    # Real loss
    real_loss = torch.log(torch.sigmoid(-real_energy_unnorm))

    # Fake loss
    fake_loss = torch.log(torch.sigmoid(fake_energys_unnorm)).sum(dim=-1)

    loss = - (real_loss + fake_loss).mean()

    return loss


@FunctionRepository.register(name='mnce')
def mnce_loss(real_energy_unnorm, fake_energys_unnorm):
    if len(real_energy_unnorm.shape) < len(fake_energys_unnorm.shape):
        real_energy_unnorm = real_energy_unnorm[..., None]

    real_energy = torch.sigmoid(real_energy_unnorm)
    fake_energys = torch.sigmoid(fake_energys_unnorm)

    whole_energy = torch.cat([real_energy, fake_energys], dim=-1)

    real_loss = real_energy
    fake_loss = torch.log(torch.exp(-whole_energy).sum(dim=-1, keepdim=True))

    loss = (real_loss + fake_loss).mean()

    return loss


def bce_score(real_energy_unnorm, fake_energys_unnorm):
    return torch.sigmoid(-real_energy_unnorm), torch.sigmoid(-fake_energys_unnorm)


def mnce_score(real_energy_unnorm, fake_energys_unnorm):
    if len(real_energy_unnorm.shape) < len(fake_energys_unnorm.shape):
        real_energy_unnorm = real_energy_unnorm[..., None]

    energy_cat = torch.cat([-real_energy_unnorm, -fake_energys_unnorm], dim=-1)
    energy_score_cat = torch.softmax(energy_cat, dim=-1)
    return energy_score_cat[..., 0], energy_score_cat[..., 1:]
