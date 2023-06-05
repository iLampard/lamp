
import torch
import torch.nn.functional as F

from ._modules import ANHPMultiHeadAttention
from ..modules import ParameterDot


class KnowEvolveANHP(torch.nn.Module):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            dim_c: int,
            dim_l: int,
            dim_d: int,
            num_layers: int,
            n_heads: int,
            dropout_rate: int,

    ):
        """

        Args:
            num_entities:
            num_relations:
            dim_c: int
                Embedding dimension of relation.
            dim_l: int
                Dimension of the hidden state.
            dim_d: int
                Embedding dimension of entity.
        """
        super(KnowEvolveANHP, self).__init__()
        self.n_e = num_entities
        self.n_r = num_relations
        self.dim_l = dim_l
        self.dim_d = dim_d
        self.dim_c = dim_c
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.activation_layer = torch.nn.Sigmoid()

        # model's state tensor buffer (keep the historical information)
        self.register_buffer('latest_entity_emb', torch.zeros(self.n_e, self.dim_d))
        # the latest relation happened to subject
        # the initial values = n_r, which will be masked in relation_emb_layer
        self.register_buffer('latest_relation_of_entity', torch.ones(self.n_e, dtype=torch.long) * self.n_r)
        self.register_buffer('latest_time_of_entity', torch.zeros(self.n_e))

        self.relation_emb_layer = torch.nn.Embedding(self.n_r + 1, embedding_dim=dim_c, padding_idx=self.n_r)

        # TODO reset m and M corresponding to dataset
        self.relation_score_layer = RelationScoreLayer(
            feat_dim=self.dim_d + self.dim_d + self.dim_c,
            dim_d=self.dim_d,
            num_layers=self.num_layers,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            time_m=1,
            time_M=2000
        )

        self.entity_update_layer = RNNEntityEmbeddingLayer(
            dim_c=dim_c,
            dim_l=dim_l,
            dim_d=dim_d
        )

        # other setting
        self.epsilon = 1e-8
        # self.intensity_activation = torch.exp
        self.intensity_activation = torch.nn.Softplus()

    def forward(
            self,
            seq_subject,
            seq_object,
            seq_time,
            seq_relation,
            init_entity_emb=None,
            init_latest_relation_of_entity=None,
            init_latest_time_of_entity=None,
            return_loss: bool = True,
            is_update_last: bool = False
    ):
        """

        Remark:
            Keep all entities (subject + object) contained in the sequence samples in the batch distinct from each
            other.

        Args:
            seq_subject: tensor with shape [B, T]
                A chronological sequence consisting of the index of the subject on which the event occurred.
            seq_object: tensor with shape [B, T]
                A chronological sequence consisting of the index of the object on which the event occurred.
            seq_time: tensor with shape [B, T]
                A chronological sequence consisting of the time at which events occurred.
            seq_relation: tensor with shape [B, T]
                A chronological sequence consisting of the index of the occurred event type.
            init_entity_emb: tensor with shape [N_e, d]
            init_latest_relation_of_entity: tensor with shape [N_e]
            init_latest_time_of_entity: tensor with shape [N_e]
            return_loss: bool
            is_update_last: bool
                If update the memory in last point. (Set True when sequence evaluation)

        Returns:

        """
        # pre-process
        batch_size, window_t = seq_subject.size()[:2]

        # convert dtype of tensor
        seq_subject = seq_subject.long()
        seq_object = seq_object.long()
        seq_relation = seq_relation.long()
        seq_time = seq_time.float()

        # Synchronize memory tensor to model's buffer when inputs of init_stats are None
        if init_entity_emb is None:
            # detach operation to remove tensor from graph
            latest_entity_emb = self.latest_entity_emb.detach()
        else:
            latest_entity_emb = init_entity_emb

        if init_latest_relation_of_entity is None:
            latest_relation_of_entity = self.latest_relation_of_entity.detach()
        else:
            latest_relation_of_entity = init_latest_relation_of_entity

        if init_latest_time_of_entity is None:
            latest_time_of_entity = self.latest_time_of_entity.detach()
        else:
            latest_time_of_entity = init_latest_time_of_entity

        # Network process
        # batch_ind = torch.arange(batch_size, dtype=torch.long)
        # shape -> [B]
        total_loss = 0
        lambda_s_o_r_t = None

        for i in range(window_t):
            # shape -> [B]
            subject_seq_i = seq_subject[:, :i + 1]
            object_seq_i = seq_object[:, :i + 1]
            relation_seq_i = seq_relation[:, :i + 1]
            time_seq_i = seq_time[:, :i + 1]

            if is_update_last:
                if i == window_t - 1:
                    update_memory = True
                else:
                    update_memory = False
            else:
                update_memory = True

            res = self.compute_intensity(
                subject_seq_i,
                object_seq_i,
                relation_seq_i,
                time_seq_i,
                init_entity_emb=latest_entity_emb,
                init_latest_relation_of_entity=latest_relation_of_entity,
                init_latest_time_of_entity=latest_time_of_entity,
                return_loss=return_loss,
                update_memory=update_memory,
            )
            if return_loss:
                lambda_s_o_r_t, loss = res
                total_loss += torch.mean(loss) / window_t
            else:
                lambda_s_o_r_t = res

        ret_tuple = (
            lambda_s_o_r_t,
            latest_entity_emb.detach(),
            latest_relation_of_entity.detach(),
            latest_time_of_entity.detach()
        )
        if return_loss:
            ret_tuple = (total_loss,) + ret_tuple
        return ret_tuple

    def inference(
            self,
            seq_subject,
            seq_object,
            seq_time,
            seq_relation,
            init_entity_emb=None,
            init_latest_relation_of_entity=None,
            init_latest_time_of_entity=None,
            predict_relation: bool = True,
            predict_object: bool = False,
    ):
        """ Predict the last point.

        Remark:
            Keep all entities (subject + object) contained in the sequence samples in the batch distinct from each
            other.

        Args:
            seq_subject: tensor with shape [B, T]
                A chronological sequence consisting of the index of the subject on which the event occurred.
            seq_object: tensor with shape [B, T]
                A chronological sequence consisting of the index of the object on which the event occurred.
            seq_time: tensor with shape [B, T]
                A chronological sequence consisting of the time at which events occurred.
            seq_relation: tensor with shape [B, T]
                A chronological sequence consisting of the index of the occurred event type.
            init_entity_emb: tensor with shape [N_e, d]
            init_latest_relation_of_entity: tensor with shape [N_e]
            init_latest_time_of_entity: tensor with shape [N_e]
            predict_relation: bool
            predict_object: bool

        Returns:

        """
        res_dict = dict()
        # link prediction
        if predict_relation:
            densities = []
            for i in range(self.n_r):
                rel = torch.ones_like(seq_relation[:, :1]) * i
                # shape -> [B]
                density_i = self.compute_conditional_density_at_now(
                    seq_subject=seq_subject,
                    seq_object=seq_object,
                    seq_relation=torch.cat([seq_relation[:, :-1], rel], dim=-1),
                    seq_time=seq_time,
                    init_entity_emb=init_entity_emb,
                    init_latest_relation_of_entity=init_latest_relation_of_entity,
                    init_latest_time_of_entity=init_latest_time_of_entity,
                )
                densities.append(density_i)
            # shape -> [B, n_r]
            res_dict['pred_relation'] = torch.stack(densities, dim=-1).detach().cpu().numpy()

        # object prediction
        if predict_object:
            densities = []
            for i in range(self.n_e):
                # shape -> [B, 1]
                obj = torch.ones_like(seq_object[:, :1]) * i
                density_i = self.compute_conditional_density_at_now(
                    seq_subject=seq_subject,
                    seq_object=torch.cat([seq_object[:, :-1], obj], dim=-1),
                    seq_relation=seq_relation,
                    seq_time=seq_time,
                    init_entity_emb=init_entity_emb,
                    init_latest_relation_of_entity=init_latest_relation_of_entity,
                    init_latest_time_of_entity=init_latest_time_of_entity,
                )
                densities.append(density_i)
            # shape -> [B, n_r]
            res_dict['pred_object'] = torch.stack(densities, dim=-1).detach().cpu().numpy()

        return res_dict

    def inference_both(
            self,
            seq_subject,
            seq_object_minus,
            seq_time,
            seq_relation_minus,
            init_entity_emb=None,
            init_latest_relation_of_entity=None,
            init_latest_time_of_entity=None,
    ):
        """ Predict the last point.

        Remark:
            Keep all entities (subject + object) contained in the sequence samples in the batch distinct from each
            other.

        Args:
            seq_subject: tensor with shape [B, T]
                A chronological sequence consisting of the index of the subject on which the event occurred.
            seq_object_minus: tensor with shape [B, T - 1]
                A chronological sequence consisting of the index of the object on which the event occurred.
            seq_time: tensor with shape [B, T]
                A chronological sequence consisting of the time at which events occurred.
            seq_relation_minus: tensor with shape [B, T - 1]
                A chronological sequence consisting of the index of the occurred event type.
            init_entity_emb: tensor with shape [N_e, d]
            init_latest_relation_of_entity: tensor with shape [N_e]
            init_latest_time_of_entity: tensor with shape [N_e]

        Returns:

        """
        res_dict = dict()
        # link prediction
        densities = []
        for i in range(self.n_r):
            rel = torch.ones_like(seq_relation_minus[:, :1]) * i
            mean_obj_idx = torch.ones_like(seq_object_minus[:, :1]) * self.n_e
            # shape -> [B]
            density_i = self.compute_conditional_density_at_now(
                seq_subject=seq_subject,
                seq_object=torch.cat([seq_object_minus, mean_obj_idx], dim=-1),
                seq_relation=torch.cat([seq_relation_minus, rel], dim=-1),
                seq_time=seq_time,
                init_entity_emb=init_entity_emb,
                init_latest_relation_of_entity=init_latest_relation_of_entity,
                init_latest_time_of_entity=init_latest_time_of_entity,
            )
            densities.append(density_i)
        # shape -> [B, n_r]
        res_dict['pred_relation'] = torch.stack(densities, dim=-1).detach().cpu().numpy()

        # object prediction
        densities = []
        for i in range(self.n_e):
            # shape -> [B, 1]
            obj = torch.ones_like(seq_object_minus[:, :1]) * i
            mean_rel_idx = torch.ones_like(seq_relation_minus[:, :1]) * self.n_r
            density_i = self.compute_conditional_density_at_now(
                seq_subject=seq_subject,
                seq_object=torch.cat([seq_object_minus, obj], dim=-1),
                seq_relation=torch.cat([seq_relation_minus, mean_rel_idx], dim=-1),
                seq_time=seq_time,
                init_entity_emb=init_entity_emb,
                init_latest_relation_of_entity=init_latest_relation_of_entity,
                init_latest_time_of_entity=init_latest_time_of_entity,
            )
            densities.append(density_i)
        # shape -> [B, n_r]
        res_dict['pred_object'] = torch.stack(densities, dim=-1).detach().cpu().numpy()

        return res_dict

    def inference_combination(
            self,
            seq_subject,
            seq_object_minus,
            seq_time,
            seq_relation_minus,
            rel_topk: int = 5,
            obj_topk: int = 20,
    ):
        """

        Remark:
            Keep all entities (subject + object) contained in the sequence samples in the batch distinct from each
            other.

        Args:
            seq_subject: tensor with shape [B, T]
                A chronological sequence consisting of the index of the subject on which the event occurred.
            seq_object_minus: tensor with shape [B, T - 1]
                A chronological sequence consisting of the index of the object on which the event occurred.
            seq_time: tensor with shape [B, T]
                A chronological sequence consisting of the time at which events occurred.
            seq_relation_minus: tensor with shape [B, T - 1]
                A chronological sequence consisting of the index of the occurred event type.

        Returns:

        """

        batch_both_pred = self.inference_both(
            seq_subject=seq_subject,
            seq_object_minus=seq_object_minus,
            seq_time=seq_time,
            seq_relation_minus=seq_relation_minus,
        )

        _, pred_relation_topk = torch.from_numpy(batch_both_pred['pred_relation']).to(
            seq_subject.device).topk(rel_topk, dim=-1)
        _, pred_object_topk = torch.from_numpy(batch_both_pred['pred_object']).to(
            seq_subject.device).topk(obj_topk, dim=-1)

        rel_obj_combination = []
        for rel_i in range(rel_topk):
            rel = pred_relation_topk[:, rel_i: rel_i + 1]
            for obj_i in range(obj_topk):
                obj = pred_object_topk[:, obj_i: obj_i + 1]
                density = self.compute_conditional_density_at_now(
                    seq_subject=seq_subject,
                    seq_object=torch.cat([seq_object_minus, obj], dim=-1),
                    seq_relation=torch.cat([seq_relation_minus, rel], dim=-1),
                    seq_time=seq_time,
                )
                rel_obj_combination.append({
                    'pred_relation': rel[:, 0].detach().cpu().numpy(),
                    'pred_object': obj[:, 0].detach().cpu().numpy(),
                    'pred_score': density.detach().cpu().numpy()
                })

        batch_both_pred['pred_rel_obj'] = rel_obj_combination
        return batch_both_pred

    def zero_states(self):
        self.latest_entity_emb.fill_(0)
        self.latest_relation_of_entity.fill_(self.n_r)
        self.latest_time_of_entity.fill_(0)

    def compute_conditional_density_at_now(
            self,
            seq_subject,
            seq_object,
            seq_relation,
            seq_time,
            num_samples: int = 10,
            init_entity_emb=None,
            init_latest_relation_of_entity=None,
            init_latest_time_of_entity=None,
    ):
        # Synchronize memory tensor to model's buffer when inputs of init_stats are None
        if init_latest_time_of_entity is None:
            latest_time_of_entity = self.latest_time_of_entity.detach()
        else:
            latest_time_of_entity = init_latest_time_of_entity

        subject_i = seq_subject[:, -1]
        object_i = seq_object[:, -1]
        time_i = seq_time[:, -1]

        # retrieve the latest relation and time for subject i
        # shape -> [B]
        latest_subject_time_i = F.embedding(subject_i, latest_time_of_entity)

        latest_object_time_i = self.get_last_object_time(object_i, latest_time_of_entity)

        # mask the delta-time of the first relation to zero
        latest_subject_time_i = latest_subject_time_i + (latest_subject_time_i <= 0) * time_i
        latest_object_time_i = latest_object_time_i + (latest_object_time_i <= 0) * time_i

        latest_couple_time = torch.maximum(latest_subject_time_i, latest_object_time_i)

        delta_time_i = time_i - latest_couple_time

        # shape -> [B]
        delta_time_dt_i = (1 / num_samples) * delta_time_i
        # shape -> [(num_samples + 1) * B]
        time_i_samples = torch.cat(
            [
                latest_couple_time + delta_time_i * j / num_samples for j in range(num_samples + 1)
            ],
            dim=0
        )

        past_seq_time = seq_time[:, :-1].repeat(num_samples + 1, 1)

        # shape -> [(num_samples + 1) * B, T]
        time_seq = torch.cat([past_seq_time, time_i_samples[:, None]], dim=-1)

        # shape -> [(num_samples + 1) * B]
        intensity_i_samples = self.compute_intensity(
            subject_seq=seq_subject.repeat(num_samples + 1, 1),
            object_seq=seq_object.repeat(num_samples + 1, 1),
            relation_seq=seq_relation.repeat(num_samples + 1, 1),
            time_seq=time_seq,
            init_entity_emb=init_entity_emb,
            init_latest_relation_of_entity=init_latest_relation_of_entity,
            init_latest_time_of_entity=init_latest_time_of_entity,
            return_loss=False,
            update_memory=False
        )
        # shape -> [num_samples + 1, B]
        intensity_i_samples = intensity_i_samples.reshape(num_samples + 1, -1)
        # shape -> [B]
        intensity_i = intensity_i_samples[-1]

        # no-event happens during [t_{i-1}, t_i]
        # shape -> [B]
        s_i = torch.exp(-torch.sum(intensity_i_samples[:-1] * delta_time_dt_i[None, :], dim=0))

        return s_i * intensity_i

    def compute_intensity(
            self,
            subject_seq,
            object_seq,
            relation_seq,
            time_seq,
            init_entity_emb=None,
            init_latest_relation_of_entity=None,
            init_latest_time_of_entity=None,
            return_loss: bool = False,
            update_memory: bool = True,
    ):
        """

        Args:
            subject_seq: tensor with shape [B, t]
                Subject index.
            object_seq: tensor with shape [B, t]
                Object index.
            relation_seq: tensor with shape [B, t]
                Index of the relation between subject and object.
            time_seq: tensor with shape [B, t]
                when the relationship occurred or will occur
            init_entity_emb: tensor with shape [N_e, d]
                Updated in process.
            init_latest_relation_of_entity: tensor with shape [N_e]
                Updated in process.
            init_latest_time_of_entity: tensor with shape [N_e]
                Updated in process.
            return_loss: bool
            update_memory: bool

        Returns:
            lambda of last point: tensor with shape [B]
            loss: tensor with shape [B]
        """
        # Synchronize memory tensor to model's buffer when inputs of init_stats are None
        if init_entity_emb is None:
            # detach operation to remove tensor from graph
            latest_entity_emb = self.latest_entity_emb.detach()
        else:
            latest_entity_emb = init_entity_emb

        if init_latest_relation_of_entity is None:
            latest_relation_of_entity = self.latest_relation_of_entity.detach()
        else:
            latest_relation_of_entity = init_latest_relation_of_entity

        if init_latest_time_of_entity is None:
            latest_time_of_entity = self.latest_time_of_entity.detach()
        else:
            latest_time_of_entity = init_latest_time_of_entity

        # setting
        subject_i = subject_seq[:, -1]
        object_i = object_seq[:, -1]
        time_i = time_seq[:, -1]
        relation_i = relation_seq[:, -1]

        # Attention Layer
        # shape -> [B, t, d]
        subject_seq_emb = F.embedding(subject_seq, latest_entity_emb.detach())
        object_seq_emb = self.get_object_emb(object_seq, latest_entity_emb)
        # shape -> [B, t, c]

        relation_seq_emb = self.get_relation_emb(relation_seq)

        # Relational Score
        # shape -> [B, t, d + d + c]
        past_seq_emb_cat = torch.cat([
            subject_seq_emb[:, :-1, :],
            object_seq_emb[:, :-1, :],
            relation_seq_emb[:, :-1, :],
        ], dim=-1)
        past_seq_time = time_seq[:, :-1]
        # shape -> [B]
        g_s_o_r = self.relation_score_layer(
            past_seq_evt_emb=past_seq_emb_cat,
            past_seq_time=past_seq_time,
            cur_evt_embs=torch.cat([
                subject_seq_emb[:, -1:, :],
                object_seq_emb[:, -1:, :],
                relation_seq_emb[:, -1:, :],
            ], dim=-1),
            cur_times=time_seq[:, -1:],
        )[:, 0]

        # Temporal Process

        latest_subject_time_i = F.embedding(subject_i, latest_time_of_entity)
        latest_object_time_i = self.get_last_object_time(object_i, latest_time_of_entity)

        # mask the delta-time of the first relation to zero
        latest_subject_time_i = latest_subject_time_i + (latest_subject_time_i <= 0) * time_i
        subject_delta_time_i = time_i - latest_subject_time_i

        latest_object_time_i = latest_object_time_i + (latest_object_time_i <= 0) * time_i
        object_delta_time_i = time_i - latest_object_time_i

        min_s_o_delta_t = torch.minimum(subject_delta_time_i, object_delta_time_i)
        # to avoid the small negative values
        min_s_o_delta_t = torch.relu(min_s_o_delta_t)

        # shape -> [B]
        epsilon = 1.0
        lambda_s_o_r_t = self.intensity_activation(g_s_o_r) * (min_s_o_delta_t + epsilon)

        # Update memory tensor
        if update_memory:
            # Dynamically Evolving Entity Representations
            # use rnn to calculate the subject and object embedding ast time_i
            latest_subject_relation_i = F.embedding(subject_seq[:, -1], latest_relation_of_entity)
            latest_subject_relation_emb_i = self.get_relation_emb(latest_subject_relation_i)

            latest_object_relation_i = F.embedding(object_seq[:, -1], latest_relation_of_entity)
            latest_object_relation_emb_i = self.get_relation_emb(latest_object_relation_i)

            # shape -> [B, d * 2 + c]
            subject_input_i = torch.cat([
                subject_seq_emb[:, -1],
                object_seq_emb[:, -1],
                latest_subject_relation_emb_i
            ], dim=-1)
            object_input_i = torch.cat([
                object_seq_emb[:, -1],
                subject_seq_emb[:, -1],
                latest_object_relation_emb_i
            ], dim=-1)
            # shape -> [B, d]
            subject_emb_i, object_emb_i = self.entity_update_layer(
                subject_input_i,
                object_input_i,
                subject_delta_time_i,
                object_delta_time_i
            )

            latest_entity_emb[subject_i] = subject_emb_i
            latest_entity_emb[object_i] = object_emb_i

            latest_relation_of_entity[subject_i] = relation_i
            latest_relation_of_entity[object_i] = relation_i

            latest_time_of_entity[subject_i] = time_i
            latest_time_of_entity[object_i] = time_i

        # Loss calculation
        if return_loss:
            batch_size = subject_i.size()[0]
            # shape -> [B]
            loss_happened = -torch.log(lambda_s_o_r_t + self.epsilon)
            # calculate the survival term use the latest subject and object embedding and current time
            # shape -> [B]
            loss_survival_correct = self.intensity_activation(
                g_s_o_r
            ) * (time_i - torch.maximum(latest_subject_time_i, latest_object_time_i)) ** 2

            loss_survival_same_subject = self.intensity_activation(
                self.relation_score_layer(
                    past_seq_evt_emb=past_seq_emb_cat,
                    past_seq_time=past_seq_time,
                    cur_evt_embs=torch.cat([
                        subject_seq_emb[:, -1:, :],
                        subject_seq_emb[:, -1:, :],
                        relation_seq_emb[:, -1:, :],
                    ], dim=-1),
                    cur_times=time_seq[:, -1:],
                )[:, 0]
            ) * (time_i - latest_subject_time_i) ** 2

            loss_survival_same_object = self.intensity_activation(
                self.relation_score_layer(
                    past_seq_evt_emb=past_seq_emb_cat,
                    past_seq_time=past_seq_time,
                    cur_evt_embs=torch.cat([
                        object_seq_emb[:, -1:, :],
                        object_seq_emb[:, -1:, :],
                        relation_seq_emb[:, -1:, :],
                    ], dim=-1),
                    cur_times=time_seq[:, -1:],
                )[:, 0]
            ) * (time_i - latest_object_time_i) ** 2

            # randomly sampling strategy to compute survival loss (all subject and object entities in the batch
            # are different)

            # Subject -> Object part (skip the Object -> Subject part for speed)
            # shape -> [B, B, d]
            sample_subject_emb_i = torch.repeat_interleave(subject_seq_emb[:, -1:, :], repeats=batch_size, dim=1)
            sample_object_emb_i = torch.repeat_interleave(object_seq_emb[None, :, -1, :], repeats=batch_size, dim=0)
            sample_relation_emb_i = torch.repeat_interleave(relation_seq_emb[:, -1:, :], repeats=batch_size, dim=1)

            # shape -> [B, B]
            survival_neg_score = self.intensity_activation(
                self.relation_score_layer(
                    past_seq_evt_emb=past_seq_emb_cat,
                    past_seq_time=past_seq_time,
                    cur_evt_embs=torch.cat([
                        sample_subject_emb_i,
                        sample_object_emb_i,
                        sample_relation_emb_i,
                    ], dim=-1),
                    cur_times=time_seq[:, -1:],
                )
            )
            # shape -> [B, B]
            survival_neg_pow_dt = torch.relu(torch.minimum(
                # t - t_o
                (time_i[:, None] - latest_object_time_i[None, :]) ** 2,
                # t - t_s
                (time_i[:, None] - latest_subject_time_i[None, :]) ** 2,
            ))

            # shape -> [B]
            loss_survival_neg = torch.sum(survival_neg_score * survival_neg_pow_dt, dim=-1)

            # Skip the Object -> Subject part for speed (just reverse the subject and object part)
            # sample_subject_emb_i = torch.repeat_interleave(object_seq_emb[:, -1:, :], repeats=batch_size, dim=1)
            # sample_object_emb_i = torch.repeat_interleave(subject_seq_emb[None, :, -1, :], repeats=batch_size, dim=0)

            loss = loss_happened + (
                    loss_survival_neg
                    - loss_survival_correct
                    - loss_survival_same_subject - loss_survival_same_object
            )
            # loss = loss_happened
            if loss.detach().cpu().numpy().mean() > 100:
                print(loss.detach().numpy().mean())
                pass

            return lambda_s_o_r_t, loss
        else:
            return lambda_s_o_r_t

    def compute_relational_score_at_now(
            self,
            seq_subject,
            seq_object,
            seq_relation,
            seq_time,
            init_entity_emb=None,
    ):
        """

        Args:
            seq_subject:
            seq_object:
            seq_relation:
            seq_time:
            init_entity_emb:
                Entity embeddings before the 'time_i'.

        Returns:

        """
        # Synchronize memory tensor to model's buffer when inputs of init_stats are None
        if init_entity_emb is None:
            # detach operation to remove tensor from graph
            latest_entity_emb = self.latest_entity_emb.detach()
        else:
            latest_entity_emb = init_entity_emb

        # Obtain the latest embeddings
        # shape -> [B, T, d]
        seq_subject_emb = F.embedding(seq_subject, latest_entity_emb.detach())
        seq_object_emb = self.get_object_emb(seq_object, latest_entity_emb)
        seq_relation_emb = self.get_relation_emb(seq_relation)

        # Relational Score
        seq_evt_emb = torch.cat([
            seq_subject_emb,
            seq_object_emb,
            seq_relation_emb,
        ], dim=-1)

        # shape -> [B]
        g_s_o_r = self.relation_score_layer(
            past_seq_evt_emb=seq_evt_emb[..., :-1, :],
            past_seq_time=seq_time[..., :-1, :],
            cur_evt_embs=seq_evt_emb[..., -1:, :],
            cur_times=seq_time[:, -1:],
        )[:, 0]
        return g_s_o_r

    def get_object_emb(self, seq_object, embedding_map=None):
        """

        Args:
            seq_object: [...]
            embedding_map: tensor with shape [n_e, dim_d]

        Returns:

        """
        if embedding_map is None:
            embedding_map = self.latest_entity_emb

        embedding_map = embedding_map.detach()
        mask = seq_object < self.n_e

        # mask the elements bigger than n_e to 0 for passing the F.embedding
        # shape -> [...]
        seq_object = seq_object * mask

        # shape -> [..., dim_d]
        origin_emb = F.embedding(seq_object, embedding_map)

        # shape -> [dim_d]
        mean_emb = torch.mean(embedding_map, dim=0)

        # shape -> [..., dim_d]
        final_emb = origin_emb * mask[..., None] + mean_emb * (~mask[..., None])

        return final_emb

    def get_relation_emb(self, seq_relation):
        """

        Args:
            seq_relation: [...]

        Returns:
            Relation embedding: [..., dim_c]
        """
        mask = seq_relation < self.n_r

        # mask the elements bigger than n_e to 0 for passing the F.embedding
        # shape -> [...]
        seq_relation = seq_relation * mask

        # shape -> [..., dim_c]
        origin_emb = self.relation_emb_layer(seq_relation)

        # shape -> [dim_c]
        mean_emb = torch.mean(self.relation_emb_layer.weight, dim=0)

        # shape -> [..., dim_c]
        final_emb = origin_emb * mask[..., None] + mean_emb * (~mask[..., None])

        return final_emb

    def get_last_object_time(self, seq_object, embedding_map=None):
        """

        Args:
            seq_object: [...]
            embedding_map: tensor with shape [n_e]

        Returns:
            Last object time: [...]
        """
        if embedding_map is None:
            embedding_map = self.latest_time_of_entity

        embedding_map = embedding_map.detach()

        mask = seq_object < self.n_e

        # mask the elements bigger than n_e to 0 for passing the F.embedding
        # shape -> [...]
        seq_object = seq_object * mask

        # shape -> [...]
        origin_time = F.embedding(seq_object, embedding_map)

        # shape -> [1]
        mean_time = torch.mean(embedding_map, dim=0)

        # shape -> [...]
        final_time = origin_time * mask + mean_time * (~mask)

        return final_time


class RNNEntityEmbeddingLayer(torch.nn.Module):
    def __init__(
            self,
            dim_c: int,
            dim_l: int,
            dim_d: int,
    ):
        """

        Args:
            dim_c: int
                Embedding dimension of relation.
            dim_l: int
                Dimension of the hidden state.
            dim_d: int
                Embedding dimension of entity.

        Returns:

        """
        super(RNNEntityEmbeddingLayer, self).__init__()
        self.activation_layer = torch.nn.Sigmoid()

        self.emb_hidden_layer = torch.nn.Linear(dim_d * 2 + dim_c, dim_l, bias=False)
        self.emb_hh_layer = torch.nn.Linear(dim_l, dim_d, bias=False)

        # subject parameters
        self.subject_emb_time_layer = torch.nn.Linear(1, dim_d, bias=False)

        # object parameters
        self.object_emb_time_layer = torch.nn.Linear(1, dim_d, bias=False)

    def forward(
            self,
            subject_input_i, object_input_i, subject_delta_time_i, object_delta_time_i
    ):
        # Subject embedding update
        # shape -> [B, l]
        subject_h = self.activation_layer(self.emb_hidden_layer(subject_input_i))
        # Equation 5
        # shape -> [B, d]
        subject_emb_i = self.activation_layer(
            self.subject_emb_time_layer(subject_delta_time_i[..., None]) + self.emb_hh_layer(subject_h)
        )

        # Object embedding update
        # shape -> [B, l]
        object_h = self.activation_layer(self.emb_hidden_layer(object_input_i))
        # Equation 6
        # shape -> [B, d]
        object_emb_i = self.activation_layer(
            self.object_emb_time_layer(object_delta_time_i[..., None]) + self.emb_hh_layer(object_h)
        )
        return subject_emb_i, object_emb_i


class TimeEmbeddingLayer(torch.nn.Module):
    def __init__(self, dim, m, M):
        super(TimeEmbeddingLayer, self).__init__()
        # shape -> [dim]
        norm = 1 / (m * torch.pow(5 * M / m, torch.arange(dim)))

        self.register_buffer('norm', norm)

    def forward(self, seq_time):
        """

        Args:
            seq_time: [...]

        Returns:
            Embedding of time: [..., dim]
        """
        return torch.sin(seq_time[..., None] * self.norm)


class RelationScoreLayer(torch.nn.Module):
    def __init__(
            self,
            feat_dim: int,
            dim_d: int,
            num_layers: int,
            n_heads: int,
            dropout_rate: int,
            time_m: int,
            time_M: int,
    ):
        super().__init__()

        self.attention_layers = torch.nn.ModuleList([
            ANHPMultiHeadAttention(
                feat_dim=feat_dim + dim_d + 1,
                hidden_dim=feat_dim,
                dropout_rate=dropout_rate,
                n_heads=n_heads,
            ) for _ in range(num_layers)
        ])
        self.time_emb_layer = TimeEmbeddingLayer(dim_d, m=time_m, M=time_M)
        self.score_out_layer = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, dim_d),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_d, 1)
        )

    def forward(
            self,
            past_seq_evt_emb,
            past_seq_time,
            cur_evt_embs,
            cur_times,
    ):
        """

        Args:
            past_seq_evt_emb: tensor with shape [..., T, feat_dim]
            past_seq_time: tensor with shape [..., T]
            cur_evt_embs: tensor with shape [..., N, feat_dim]
            cur_times: tensor with shape [..., N]

        Returns:
            Score for each cur_evt_emb: tensor with shape [..., N]
        """
        # shape -> [..., T, dim_d]
        past_seq_time_emb = self.time_emb_layer(past_seq_time)
        # shape -> [..., N, dim_d]
        cur_time_embs = self.time_emb_layer(cur_times)

        T = past_seq_evt_emb.size()[-2]
        if T == 0:
            # shape -> [..., N]
            scores = self.score_out_layer(cur_evt_embs)[..., 0]
            return scores

        for att_layer in self.attention_layers:
            # shape -> [..., T, feat_dim + dim_d + 1]
            past_e_i = torch.cat(
                [
                    past_seq_evt_emb,
                    past_seq_time_emb,
                    torch.ones_like(past_seq_evt_emb[..., :1])
                ],
                dim=-1
            )
            # shape -> [..., N, feat_dim + dim_d + 1]
            cur_e_is = torch.cat(
                [
                    cur_evt_embs,
                    cur_time_embs,
                    torch.ones_like(cur_evt_embs[..., :1])
                ],
                dim=-1
            )

            whole_evt_emb = att_layer(
                queries=torch.cat([past_e_i, cur_e_is], dim=-2),
                keys=past_e_i,
                values=past_e_i,
            )
            # update event_emb
            past_seq_evt_emb = past_seq_evt_emb + torch.tanh(whole_evt_emb[..., :T, :])
            cur_evt_embs = cur_evt_embs + torch.tanh(whole_evt_emb[..., T:, :])

        # shape -> [..., N]
        scores = self.score_out_layer(cur_evt_embs)[..., 0]
        return scores
