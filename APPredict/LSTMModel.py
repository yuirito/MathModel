import torch.nn as nn
import torch

class lstm(nn.Module):

    def __init__(self, input_size=11, hidden_size=33, num_layers=3 , output_size=18 , dropout=0.1, batch_first=True):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, bidirectional=False,hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 4*self.output_size),
            nn.GELU(),
            nn.Linear(4 * self.output_size, self.output_size),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):

        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))


        out = self.mlp(hidden)

        return out,hidden
class lstm_2(nn.Module):
    def __init__(self,input_size=8, hidden_size=32, num_layers=3 , output_size=18 ,first_size=63, dropout=0, batch_first=True):
        super(lstm_2, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.first_size = first_size
        self.linear_1 = nn.Linear(self.hidden_size+self.first_size, self.output_size)

    def forward(self,x_act,x_pre,pre_model):
        out,h = pre_model(x_act)
        h = h[-1, :, :]
        h.squeeze(1)
        x = torch.cat((h,x_pre),1)
        out = self.linear_1(x)
        return out


class cnn_lstm(nn.Module):

    def __init__(self, input_size=11, hidden_size=33, num_layers=3 , output_size=18 , dropout=0.1, batch_first=True):
        super(cnn_lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.conv = nn.Conv2d(in_channels=72, out_channels=72, kernel_size=(4, 1), stride=1)
        self.rnn = nn.LSTM(input_size=self.input_size, bidirectional=False,hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 4*self.output_size),
            nn.GELU(),
            nn.Linear(4 * self.output_size, self.output_size),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        x=self.conv(x)
        x=x.squeeze(2)
        #print(x.shape)

        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size

        out = self.mlp(hidden)

        return out,hidden

class cnn_lstm_2(nn.Module):
    def __init__(self,input_size=8, hidden_size=32, num_layers=3 , output_size=18 ,first_size=63, dropout=0, batch_first=True):
        super(cnn_lstm_2, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.first_size = first_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 1), stride=1)
        self.linear_1 = nn.Linear(self.hidden_size+self.first_size, self.output_size)

    def forward(self,x_act,x_pre,pre_model):
        out,h = pre_model(x_act)
        h = h[-1, :, :]
        h.squeeze(1)
        #print("x_preshape:")
        x_pre=self.conv(x_pre)

        x_pre=x_pre.squeeze(2)
        x_pre = x_pre.squeeze(1)
        #print(x_pre.shape)
        x = torch.cat((h,x_pre),1)
        out = self.linear_1(x)
        return out


