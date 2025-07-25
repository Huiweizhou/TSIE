import torch
import torch.nn as nn
import torch.nn.functional as F
from Sage import SAGE
import math
import dgl
import numpy as np
import random
from operator import itemgetter


class tsie(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, out_dim, rnn_wnd,
            attn_wnd, time_steps, aggr, dropout_p,
            nhead, num_layers, nhead_2, num_layers_2, dropout_trans, device
    ):
        super(tsie, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.time_steps = time_steps
        self.attn_wnd = attn_wnd if attn_wnd != -1 else time_steps
        self.device = device
        self.rnn_wnd = rnn_wnd

        self.masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=1).to(device)
        re_masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=self.attn_wnd)
        self.masks += re_masks.transpose(0, 1).to(device)

        self.gru1 = nn.ModuleList([
            GRUModel(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout_rate=0.1).to(device)
            for _ in range(1)
        ])
        self.gru2 = nn.ModuleList([
            GRUModel(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout_rate=0.1).to(device)
            for _ in range(1)
        ])

        self.transformer = nn.ModuleList([
            TransformerEncoder(attention_dim=hidden_dim * 2, num_heads=nhead, dropout=dropout_trans, mask=self.masks).to(device)
            for _ in range(num_layers)
        ])
        self.transformer2 = nn.ModuleList([
            TransformerEncoder(attention_dim=hidden_dim * 2, num_heads=nhead_2, dropout=dropout_trans, mask=self.masks).to(device)
            for _ in range(num_layers_2)
        ])

        self.fusion_layer = GatedAddition(input_dim=hidden_dim * 2)

        self.out = nn.Linear(hidden_dim * 2, out_dim, bias=False)
        self.Sage = SAGE(input_dim, hidden_dim, aggr, dropout_p)

    def forward(self, data_list_i, data_list_j):
        i_temporal_embeddings = []
        j_temporal_embeddings = []
        for t in range(self.time_steps):
            x_i = data_list_i[t][2][0].srcdata['feats']
            x_j = data_list_j[t][2][0].srcdata['feats'] 
            sage_o_i = self.Sage(data_list_i[t][2], x_i) 
            sage_o_j = self.Sage(data_list_j[t][2], x_j)
            sage_o_i = F.normalize(sage_o_i, p=2, dim=1)
            sage_o_j = F.normalize(sage_o_j, p=2, dim=1)
            _, idx_i = torch.unique(data_list_i[t][1], return_inverse=True)
            _, idx_j = torch.unique(data_list_j[t][1], return_inverse=True)

            i_temporal_embeddings.append(sage_o_i[idx_i])
            j_temporal_embeddings.append(sage_o_j[idx_j])

        i_temporal_embeddings = torch.stack(i_temporal_embeddings)  # [num_steps, bs, hiddendim]
        j_temporal_embeddings = torch.stack(j_temporal_embeddings)
        # [time_steps, batchsize, hiddendim]
        u_temporal_embeddings = i_temporal_embeddings
        v_temporal_embeddings = j_temporal_embeddings

        u_v_difference_embeddings = []
        v_u_difference_embeddings = []
        u_history_embeddings = []
        v_history_embeddings = []

        for i in range(self.time_steps):
            u_to_abs = u_temporal_embeddings[i, :, :].unsqueeze(0).expand(self.time_steps, -1, -1)
            v_to_abs = v_temporal_embeddings[i, :, :].unsqueeze(0).expand(self.time_steps, -1, -1)
            u_v_diff = v_to_abs - u_temporal_embeddings
            v_u_diff = u_to_abs - v_temporal_embeddings

            u_v_diff = u_v_diff[:i + 1, :, :]
            v_u_diff = v_u_diff[:i + 1, :, :]

            if i >= self.rnn_wnd:
                v_u_diff = v_u_diff[(i + 1 - self.rnn_wnd):i + 1, :, :]
                u_v_diff = u_v_diff[(i + 1 - self.rnn_wnd):i + 1, :, :]

            for rnn in self.gru1: 
                v_u = rnn(v_u_diff)[-1]  # [bs, hiddendim]
                u_v = rnn(u_v_diff)[-1]

            u_his = u_temporal_embeddings[:i + 1, :, :]
            v_his = v_temporal_embeddings[:i + 1, :, :]

            if i >= self.rnn_wnd:
                u_his = u_his[(i + 1 - self.rnn_wnd):i + 1, :, :]
                v_his = v_his[(i + 1 - self.rnn_wnd):i + 1, :, :]

            for rnn in self.gru2:
                u_his = rnn(u_his)[-1]
                v_his = rnn(v_his)[-1]

            u_v_difference_embeddings.append(u_v)
            v_u_difference_embeddings.append(v_u)
            u_history_embeddings.append(u_his)
            v_history_embeddings.append(v_his)

        u_v_difference_embeddings = torch.stack(u_v_difference_embeddings)
        v_u_difference_embeddings = torch.stack(v_u_difference_embeddings)
        u_history_embeddings = torch.stack(u_history_embeddings)
        v_history_embeddings = torch.stack(v_history_embeddings)

        uv_embeddings = torch.cat([v_u_difference_embeddings, u_v_difference_embeddings], dim=-1)  # [num_steps, bs, 2*hiddendim]
        outs = uv_embeddings
        his = torch.cat([u_history_embeddings, v_history_embeddings], dim=-1)

        for trans in self.transformer:
            outs = trans(outs, outs, outs)  # [num_steps, bs, 2*hiddendim]
        for trans in self.transformer2:
            his = trans(his, his, his)  # [num_steps, bs, 2*hiddendim]

        outs = self.fusion_layer(outs, his)

        return self.out(outs).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.act(out)
        out = self.dropout(out)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1, mask: torch.Tensor = None):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.mask = mask

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None):
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        hidden_states, _ = self.multi_head_attention(query=inputs_query, key=inputs_key,
                                                     value=inputs_value, attn_mask=self.mask)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs


class GatedAddition(nn.Module):
    def __init__(self, input_dim):
        super(GatedAddition, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, main, aux):

        gate_value = torch.sigmoid(self.gate(main))

        main_transformed = main
        aux_transformed = aux
        main_transformed = torch.relu(main_transformed)
        aux_transformed = torch.relu(aux_transformed)

        output = gate_value * main_transformed + (1 - gate_value) * aux_transformed

        return output
