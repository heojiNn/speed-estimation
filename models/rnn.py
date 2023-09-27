import audobject
import torch
import torch.nn as nn
import math

from models.util import init_layer, init_bn

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# ========================================= RNNS ==========================================================
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

""" SELFMADE """


class GRU(nn.Module, audobject.Object):
    def __init__(
            self,
            input_size=500 * 64,
            hidden_size=256,
            num_layers=1,
            device='cnn:0',
            dropout_p=0.2,
            bidirectional=False
    ):
        super(GRU, self).__init__()
        # members
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        # layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers, dropout=dropout_p, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size, 1)
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, H * W]
        x, _ = self.gru(x)
        x = self.linear(x)  # [B, H * W] -> [B]
        return x


# == DNN
class LSTM_deprecated(nn.Module, audobject.Object):
    def __init__(
            self,
            input_size=500 * 64,
            hidden_size=256,
            num_layers=1,
            device='cnn:0',
            dropout_p=0.2,
            batch_norm=False,
            bidirectional=False,
    ):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.bidirectional = bidirectional

        # augmentations:
        self.spec_augmenter = data_augment.SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        )  # 2 2
        self.filter_augmenter = data_augment.FilterAugment()

        # layers:
        if batch_norm:
            self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, 1)
        if bidirectional:
            self.fc = nn.Linear(2 * hidden_size, 1)

        self.init_weights()

    def forward(self, x):
        # x = self.filter_augmenter(x)
        x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, H * W]
        if self.batch_norm:
            x = self.bn(x)
        lstm_output, _ = self.lstm(x)
        out = self.fc(lstm_output)
        return out

    def init_weights(self):
        init_layer(self.fc)
        if self.batch_norm:
            init_bn(self.bn)


class LSTM2(nn.Module, audobject.Object):
    def __init__(
            self,
            input_size=500 * 64,
            hidden_size=256,
            num_layers=1,
            device='cnn:0',
            dropout_p=0.2,
            batch_norm=False,
            bidirectional=False,
    ):
        super(LSTM2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.bidirectional = bidirectional

        # i_t
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # f_t
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # c_t
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # if batch_norm:
        #     self.bn = nn.BatchNorm1d(input_size)
        #
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
        #                     batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        #
        # self.fc = nn.Linear(hidden_size, 1)
        #
        # if bidirectional:
        #     self.fc = nn.Linear(2 * hidden_size, 1)

        self.init_weights()

    def forward(self, x):
        # INPUT: (B, sequence_length, feature_length)
        # x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, H * W]
        x = x.squeeze(1)  # [B,C,H,W] -> [B, H, W]

        if self.batch_norm:
            x = self.bn(x)
        lstm_output, _ = self.lstm(x)
        out = self.fc(lstm_output)#.mean(1)
        return out

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTM(nn.Module, audobject.Object):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            device,
            dropout_p=0.2,
            batch_norm=False,
            num_classes=1,
            bidirectional = False
    ):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.bidirectional = bidirectional

        if batch_norm:
            self.bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_p, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, num_classes)

        if bidirectional:
            self.fc = nn.Linear(2 * hidden_size, 1)

        self.init_weights()

    def forward(self, x):
        x = x.squeeze(1)  # [B, C, H, W] - > [B, H, W]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Decode the hidden state of the last time step
        return out

    def init_weights(self):
        init_layer(self.fc)
        if self.batch_norm:
            init_bn(self.bn)


if __name__ == '__main__':

    input_size = 64
    hidden_size = 128
    num_layers = 2
    num_classes = 1

    mdl = LSTM(
            input_size=64,
            hidden_size=500,
            num_layers=2,
            device='cnn:0',
            dropout_p=0.2,
            batch_norm=False,
            bidirectional=False,
    )

    batch = torch.randn([16, 1, 500, 64])
    preds = mdl(batch)

    print(preds.shape)
    print(preds)


