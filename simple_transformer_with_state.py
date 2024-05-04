# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University
import pdb

import torch
from torch import nn


class TF_RNN_Past_State(nn.Module):
    'size_s is the output size'
    def __init__(
        self,
        input_size_imu, size_s,
        rnn_hid_size,
        tf_hid_size, tf_in_dim, n_heads, tf_layers,
        dropout, in_dropout, past_state_dropout,
        with_rnn=True,
        with_acc_sum=False
    ):
        super(TF_RNN_Past_State, self).__init__()

        if with_acc_sum:
            print("model with acc sum")
            self.in_linear = nn.Linear(input_size_imu + size_s + 18, tf_in_dim)          # TODO: hardcoded
        else:
            self.in_linear = nn.Linear(input_size_imu + size_s, tf_in_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=tf_in_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=tf_hid_size)
        'You may want to check ths shape of tf_encode (its output size) here'
        self.tf_encode = torch.nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        # (len, bs, input_size_x)

        self.with_rnn = with_rnn
        self.num_layers = 2
        self.is_bidirectional = True
        self.bidirectional_scaler=2 if self.is_bidirectional else 1
        if with_rnn:
            self.lstm = nn.LSTM(input_size=tf_in_dim,
                                hidden_size=rnn_hid_size,
                                num_layers=self.num_layers,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=self.is_bidirectional)
            # 'Well, here you select batch_first to True. Make sure you have the right shape'
            # self.rnn = torch.nn.RNN(input_size=tf_in_dim,
            #                         hidden_size=rnn_hid_size,
            #                         num_layers=1,
            #                         nonlinearity='tanh',
            #                         batch_first=True,
            #                         dropout=dropout,
            #                         bidirectional=False)        # UNI-directional

            self.linear = nn.Linear(rnn_hid_size*self.bidirectional_scaler, size_s)
        else:
            print("no RNN layer")
            self.rnn = None
            self.linear = nn.Linear(tf_in_dim, size_s)

        self.rnn_hid_size = rnn_hid_size
        self.in_dropout = in_dropout
        self.n_heads = n_heads
        self.past_state_dropout = past_state_dropout

        # print("no c in input")
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        'I do not really get this..'
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, x_imu, x_s):
        device = 'cuda' if x_imu.get_device() >= 0 else None

        x_imu = x_imu.clone()
        x_s = x_s.clone()
        'This nan solution is important'
        x_s[x_s.isnan()] = 0.0        # if include dip data, could be nan
        bs = x_imu.size()[0]
        seq_len = x_imu.size()[1]

        # x_imu_r = nn.Dropout(self.imu_dropout_R)(x_imu[:, :, :6*9])
        # x_imu_acc = nn.Dropout(self.imu_dropout_acc)(x_imu[:, :, 6*9:6*9+18])
        # x_imu = torch.cat((x_imu_r, x_imu_acc), dim=2)

        x_imu = (nn.Dropout(self.in_dropout))(x_imu)
        # exclude root info in history input
        'This is important, the root information is ignored~!'
        'You ignore that information by putting its value to 0.0'
        x_s[:, :, 18*6: 18*6 + 3] *= 0.0
        # x_s[:, :, 18*6:] *= 0.0
        'so, x_s is your past state?! I think its the past output/ground truth..? yes.'
        x_s = (nn.Dropout(self.past_state_dropout))(x_s)
        x = torch.cat((x_imu, x_s), dim=2)
        x = self.in_linear(x)

        # # x shape (b, t, e)
        # x = x.permute(1, 0, 2)
        # # (len, bs, input_size_x)
        #
        # 'We need to double check this...?'
        # # mask future state and IMU
        # mask = self._generate_square_subsequent_mask(len(x)).to(device)
        #
        # # TODO: does not know if useful, should not harm
        # x = x.reshape(seq_len, bs, self.n_heads, -1)
        # x = x.transpose(2, 3).reshape(seq_len, bs, -1)
        #
        # x = self.tf_encode(x, mask)
        # # (len, bs, input_size_x)
        # x = torch.transpose(x, 0, 1)

        if self.with_rnn:
            # x shape (bs, L, input_size(H_in))
            h0 = torch.zeros(self.num_layers*self.bidirectional_scaler, x.size()[0], self.rnn_hid_size).to(device)  # Initialize hidden state
            c0 = torch.zeros(self.num_layers*self.bidirectional_scaler, x.size()[0], self.rnn_hid_size).to(device)  # Initialize cell state

            x, _ = self.lstm(x, (h0, c0))  # Forward pass through the LSTM layer
            # # init hidden state
            # hidden = torch.zeros(1, x.size()[0], self.rnn_hid_size).to(device=device)
            # pdb.set_trace()
            # x, _ = self.rnn(x, hidden)
            # # x shape (bs, L, self.emd_size(H_out) * 2)
        return self.linear(x)
