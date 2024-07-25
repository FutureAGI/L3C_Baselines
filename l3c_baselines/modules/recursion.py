import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, cache=None, need_cache=False):
        if(cache is None):
            lstm_out, new_cache = self.lstm(src)
        else:
            lstm_out, new_cache = self.lstm(src, cache)
        output = self.fc(lstm_out)
        return output, new_cache

class PRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Plastic Parameters
        self.Wih_a = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.Wih_b = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.Wih_c = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.Wih_d = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.bih = nn.Parameter(torch.Tensor(hidden_size))

        # Dopamine gates
        self.gates = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.Sigmoid(),
        )

        # 初始化参数
        self.reset_parameters()
        self.alpha = 1.0e-3
        self.beta = 1.0e-2

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wih_a)
        nn.init.xavier_uniform_(self.Wih_b)
        nn.init.xavier_uniform_(self.Wih_c)
        nn.init.xavier_uniform_(self.Wih_d)
        nn.init.zeros_(self.bih)

    def init_memories(self, batch_size, device):
        wih = torch.zeros(batch_size, self.hidden_size, self.input_size + self.hidden_size)
        h = torch.zeros(batch_size, self.hidden_size)
        return h.to(device), wih.to(device)

    def forward(self, inputs, memories=None):
        if(memories is not None):
            hidden, wih = memories
        else:
            hidden, wih = self.init_memories(inputs.shape[0], inputs.device)

        x = torch.cat((inputs, hidden), 1)

        gate_signal = self.gates(x)
        forget_gate = gate_signal[:, :self.hidden_size]
        input_gate = gate_signal[:, self.hidden_size:]

        new_hidden = F.tanh(torch.einsum('ij,ikj->ik', x, wih) + self.bih)

        x = x.unsqueeze(1)
        x_ = torch.ones_like(x).to(inputs.device)
        y = new_hidden.unsqueeze(2)
        y_ = torch.ones_like(y).to(inputs.device)

        wih = wih * (1.0 - self.beta * forget_gate.unsqueeze(2)) + self.alpha * input_gate.unsqueeze(2) * (torch.bmm(y, x) * self.Wih_a +
                torch.bmm(y_, x) * self.Wih_b +
                torch.bmm(y, x_) * self.Wih_c +
                torch.bmm(y_, x_) * self.Wih_d)

        return new_hidden, wih

class PRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PRNN, self).__init__()
        self.hidden_size = hidden_size
        self.prnn_cell = PRNNCell(input_size, hidden_size) 
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, cache=None, need_cache=False):
        if(cache is None):
            h, W = self.prnn_cell.init_memories(src.shape[0], src.device)
        else:
            h, W = cache
        hiddens = []
        for i in range(src.shape[1]):
            h, W = self.prnn_cell(src[:, i], (h, W))
            hiddens.append(h.unsqueeze(1))
        outputs = self.fc(torch.cat(hiddens, dim=1))
        new_cache = (h, W)
        return outputs, new_cache

class WrapperMer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, fc_hidden_size, temporal_module, dropout=0.1):
        super(WrapperMer, self).__init__()
        self.temporal_encoder = temporal_module(hidden_size, inner_hidden_size, hidden_size)

        self.linear1 = nn.Linear(hidden_size, fc_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(fc_hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size, eps=1.0e-5)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1.0e-5)

        self.activation = nn.GELU()

    def forward(self, src, cache=None, need_cache=False):
        # Residual Connection
        norm_src = self.norm1(src)
        outputs, cache = self.temporal_encoder(norm_src, cache=cache, need_cache=need_cache)

        outputs = outputs + src

        # FeedForward + Residual
        outputs = outputs + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(outputs))))))

        return outputs, cache

class MemoryLayers(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, fc_hidden_size, temporal_module, num_layers, dropout=0.1):
        super(MemoryLayers, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([WrapperMer(hidden_size, inner_hidden_size, fc_hidden_size, temporal_module, dropout=dropout) for _ in range(self.num_layers)])

    def forward(self, src, cache=None, need_cache=False, checkpoints_density=-1):
        # Residual Connection
        if(need_cache):
            new_cache = []
        else:
            new_cache = None

        output = src

        for i, layer in enumerate(self.layers):
            if(cache is None):
                l_cache = None
            else:
                l_cache = cache[i]
            output, n_cache = layer(output, cache=l_cache, need_cache=True)
            if(need_cache):
                new_cache.append(n_cache)

        return output, new_cache

if __name__=='__main__':
    inputs = torch.randn(4, 8, 64)
    model = MemoryLayers(64, 64, 256, SimpleLSTM, 3)
    outputs, mems = model(inputs, need_cache=True)
    print(outputs.shape, mems[0][0].shape, mems[0][1].shape, mems[1][0].shape, mems[1][1].shape)
