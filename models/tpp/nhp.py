from .base_model import BaseModel

import torch
from torch import nn


class ContTimeLSTMCell(nn.Module):
    """  LSTM Cell in Neural Hawkes Model    """

    def __init__(self, hidden_dim, beta=1.0, device=None):
        """ Initialize the continuous LSTM cell  """
        super(ContTimeLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device(device or 'cpu')
        self.init_dense_layer(hidden_dim, bias=True, beta=beta)

    def init_dense_layer(self, hidden_dim, bias, beta):
        """  Initialize linear layers given equations (5a-6c) in the paper  """
        self.layer_input = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_forget = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_output = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_input_bar = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_forget_bar = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_pre_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_decay = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=bias),
            nn.Softplus(beta=beta))

    def forward(self, x_i, hidden_i_minus, cell_i_minus, cell_bar_i_minus_1):
        """
        :param x_i: input
        :param hidden_i_minus: hidden state at t_i-
        :param cell_i_minus: cell state at t_i-
        :param cell_bar_i_minus_1: cell bar state at t_{i-1}
        :return: cell state, cell bar state, decay and output at t_i
        """

        x_i_ = torch.cat((x_i, hidden_i_minus), dim=1).float()

        # update input gate - equation (5a)
        gate_input = torch.sigmoid(self.layer_input(x_i_))

        # update forget gate - equation (5b)
        gate_forget = torch.sigmoid(self.layer_forget(x_i_))

        # update output gate - equation (5d)
        gate_output = torch.sigmoid(self.layer_output(x_i_))

        # update input bar - similar to equation (5a)
        gate_input_bar = torch.sigmoid(self.layer_input_bar(x_i_))

        # update forget bar - similar to equation (5b)
        gate_forget_bar = torch.sigmoid(self.layer_forget_bar(x_i_))

        # update gate z - equation (5c)
        gate_pre_c = torch.tanh(self.layer_pre_c(x_i_))

        # update gate decay - equation (6c)
        gate_decay = self.layer_decay(x_i_)

        # update cell state to t_i+ - equation (6a)
        cell_i = gate_forget * cell_i_minus + gate_input * gate_pre_c

        # update cell state bar - equation (6b)
        cell_bar_i = gate_forget_bar * cell_bar_i_minus_1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime):
        """ Cell and hidden state decay  """
        c_t = cell_bar_i + (cell_i - cell_bar_i) * \
              torch.exp(-gate_decay * dtime)
        h_t = gate_output * torch.tanh(c_t)

        return c_t, h_t


class NHP(BaseModel):
    """ Neural Hawkes Model with RNN Cells   """

    def __init__(self, model_config):
        super(NHP, self).__init__(model_config)
        self.beta = model_config.get('beta', 1.0)
        self.bias = model_config.get('bias', False)
        self.device = model_config.get('device', 'cpu')
        self.rnn_cell = ContTimeLSTMCell(self.hidden_dim)

        self.layer_intensity = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_event_types_no_pad, self.bias),
            nn.Softplus(self.beta))

    def init_state(self, batch_size):
        """ Initialize hidden and cell states for each run  """
        h_t, c_t, c_bar = torch.zeros(batch_size,
                                      3 * self.hidden_dim,
                                      device=self.device).chunk(3, dim=1)
        return h_t, c_t, c_bar

    def forward(self, batch, **kwargs):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _, type_mask = batch

        all_hiddens = []
        all_outputs = []
        all_cells = []
        all_cell_bars = []
        all_decays = []

        steps = kwargs.get('steps', None)

        # last event has no time label
        max_seq_length = steps if steps is not None else event_seq.size(1) - 1

        batch_size = len(event_seq)
        h_t, c_t, c_bar_i = self.init_state(batch_size)

        # if only one event, then we dont decay
        if max_seq_length == 1:
            types_sub_batch = event_seq[:, 0]
            x_t = self.layer_event_emb(types_sub_batch)
            cell_i, c_bar_i, decay_i, output_i = \
                self.rnn_cell(x_t, h_t, c_t, c_bar_i)

            # Append all output
            all_outputs.append(output_i)
            all_decays.append(decay_i)
            all_cells.append(cell_i)
            all_cell_bars.append(c_bar_i)
            all_hiddens.append(h_t)
        else:
            # Loop over all events
            for i in range(max_seq_length):
                dt = time_delta_seq[:, i + 1]  # need to carefully check here
                types_sub_batch = event_seq[:, i]
                x_t = self.layer_event_emb(types_sub_batch)

                # cell_i  (batch_size, process_dim)
                cell_i, c_bar_i, decay_i, output_i = \
                    self.rnn_cell(x_t, h_t, c_t, c_bar_i)

                # States decay - Equation (7) in the paper
                c_t, h_t = self.rnn_cell.decay(cell_i,
                                               c_bar_i,
                                               decay_i,
                                               output_i,
                                               dt[:, None])

                # Append all output
                all_outputs.append(output_i)
                all_decays.append(decay_i)
                all_cells.append(cell_i)
                all_cell_bars.append(c_bar_i)
                all_hiddens.append(h_t)

        # (batch_size, max_seq_length, hidden_dim)
        cell_stack = torch.stack(all_cells, dim=1)
        cell_bar_stack = torch.stack(all_cell_bars, dim=1)
        decay_stack = torch.stack(all_decays, dim=1)
        output_stack = torch.stack(all_outputs, dim=1)

        # shape -> (batch_size, max_seq_length, hidden_dim)
        hiddens_stack = torch.stack(all_hiddens, dim=1)
        # (batch_size, max_seq_length, 4, hidden_dim)
        decay_states_stack = torch.stack((cell_stack,
                                          cell_bar_stack,
                                          decay_stack,
                                          output_stack),
                                         dim=2)

        return hiddens_stack, decay_states_stack

    def compute_loglik(self, batch):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _, type_mask = batch

        hiddens_ti, decay_states = self.forward(batch)

        # Num of samples in each batch and num of event time point in the sequence
        num_sample, num_times, _ = hiddens_ti.size()

        # Lambda(t) right before each event time point
        # lambda_at_event - (batch_size, num_times, num_event_types)
        lambda_at_event = self.layer_intensity(hiddens_ti.float())

        # Retrieve the subtype intensities by masking
        # (num_samples, num_times, num_event_types) no padding
        lambda_type_mask = type_mask[:, 1:]

        # Sum of lambda over every type and every event point
        # term 1 in Equation (8)
        # (num_samples, num_times)
        event_lambdas = torch.sum(lambda_at_event * lambda_type_mask, dim=2) + self.eps

        event_lambdas.masked_fill_(~batch_non_pad_mask[:, 1:], 1.0)

        # (num_samples, num_times)
        event_ll = torch.log(event_lambdas)

        # Compute the big lambda integral in equation (8)
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length
        num_mc_sample = self.num_steps_integral_loss

        # interval_t_sample - (batch_size, num_times, num_mc_sample, 1)
        # for every batch and every event point => do a sampling (num_mc_sampling, 1)
        interval_t_sample = torch.rand(num_sample, num_times, num_mc_sample, 1, device=self.device)
        interval_t_sample = time_delta_seq[:, 1:, None, None] * interval_t_sample  # size expansion

        # get all decay states
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, decays, outputs = decay_states.unbind(dim=-2)

        # Use broadcasting to compute the decays at all time steps
        # at all sample points
        # h_ts shape (batch_size, num_times, num_mc_sample, hidden_dim)
        # cells[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
        _, h_ts = self.rnn_cell.decay(cells[:, :, None, :],
                                      cell_bars[:, :, None, :],
                                      decays[:, :, None, :],
                                      outputs[:, :, None, :],
                                      interval_t_sample)

        # lambda_t_sample - (batch_size, num_times, num_mc_sample, process_dim)
        lambda_t_sample = self.layer_intensity(h_ts.float())

        # total_lambda_sample - (batch_size, num_times, n_mc_sample)
        total_lambda_sample = lambda_t_sample.sum(dim=-1)

        # mask for the lambda(t) - (batch_size, num_times, 1)
        lambda_mask = lambda_type_mask.sum(dim=2)

        # interval_integral - (batch_size, num_times)
        # interval_integral = length_interval * average of sampled lambda(t)
        interval_integral = time_delta_seq[:, 1:] * \
                            total_lambda_sample.mean(dim=2) * lambda_mask

        # Euler sum of the integral
        non_event_ll = interval_integral

        # (num_samples, num_times)
        return event_ll, non_event_ll, lambda_at_event

    def compute_intensities_at_sampled_times(self, event_seq, time_seq, sampled_times):
        # Assumption: all the sampled times are distributed [time_seq[...,-1], next_event_time]
        # used for thinning algorithm
        num_batches = event_seq.size(0)
        seq_len = event_seq.size(1)
        assert num_batches == 1, "Currently, no support for batch mode (what is a good way to do batching in thinning?)"
        # if num_batches == 1 and num_batches < sampled_times.size(0):
        #     _sample_size = sampled_times.size(0)
        #     # multiple sampled_times
        #     event_seq = event_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
        #     time_seq = time_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
        #     num_batches = event_seq.size(0)
        assert (time_seq[:, -1:] <= sampled_times).all(), "sampled times must occur not earlier than last events!"

        time_delta_seq = time_seq[:, 1:] - time_seq[:, :-1]

        time_delta_seq = torch.cat([torch.zeros_like(time_seq[:, :1]), time_delta_seq],
                                   dim=-1)
        input_ = time_seq, time_delta_seq, event_seq, None, None, None

        # forward to the last but one event
        hiddens_ti, decay_states = self.forward(input_, steps=seq_len if seq_len == 1 else None)

        # update the states given last event
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, decays, outputs = decay_states.unbind(dim=-2)

        # if there is only one event in the sequence, we directly decay it to sample times
        # else we update the RNN by inserting the last event
        if seq_len == 1:
            cell_i = cells[:, -1]
            c_bar_i = cell_bars[:, -1]
            decay_i = decays[:, -1]
            output_i = outputs[:, -1]
        else:
            x_t = self.layer_event_emb(event_seq[:, -1])
            h_t = hiddens_ti[:, -1]
            c_t = cells[:, -1]
            c_bar_t = cell_bars[:, -1]
            cell_i, c_bar_i, decay_i, output_i = \
                self.rnn_cell(x_t, h_t, c_t, c_bar_t)

        # Use broadcasting to compute the decays at all time steps
        # at all sample points
        # h_ts shape (batch_size, num_samples, hidden_dim)
        _, h_ts = self.rnn_cell.decay(cell_i[:, None, :],
                                      c_bar_i[:, None, :],
                                      decay_i[:, None, :],
                                      output_i[:, None, :],
                                      sampled_times[:, :, None])

        sampled_intensities = self.layer_intensity(h_ts)

        return sampled_intensities