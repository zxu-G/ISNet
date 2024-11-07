import torch
import math
import torch.nn as nn
from torch.nn import MultiheadAttention



class AddNorm(nn.Module):

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):         # hidden_dim  32
        super().__init__()
        self.multi_head_self_attention  = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout                    = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA   = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA   = self.dropout(hidden_states_MSA)
        return hidden_states_MSA



class Delay_aware(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, **model_args):
        super().__init__()
        self.feature_args = model_args['feature_args']
        self.observe = ['MgII', 'SYMH', 'ASYH', 'Flow_Speed']
        self.dropout = model_args['dropout']
        self.normalized_shape = [model_args['seq_length'], model_args['feature_emb_dim']]

        for feature in self.observe:
            if self.feature_args[feature]:
                setattr(self, f"attention_layer_{feature}",
                        AttentionLayer(hidden_dim, num_heads, model_args['dropout'], bias))
                setattr(self, f"addnorm_{feature}",
                        AddNorm(self.normalized_shape, self.dropout))

        self.pos_encoder    = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.attention_layer = AttentionLayer(hidden_dim, num_heads, model_args['dropout'], bias)




    def forward(self, history_data, Feature_feat):

        # reshape
        [batch_size, seq_len, num_nodes, hidden_dim]    = history_data.shape
        history_data   = history_data.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        history_data   = self.pos_encoder(history_data)

        Observed = {}

        for feature in self.observe:
            if self.feature_args[feature]:
                # print(feature)
                Feature_feat[feature] = Feature_feat[feature].transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
                Feature_feat[feature] = self.pos_encoder(Feature_feat[feature])
                attention_feat = getattr(self, f'attention_layer_{feature}')(history_data, Feature_feat[feature], Feature_feat[feature])
                Feature_feat[feature] = getattr(self, f'addnorm_{feature}')(Feature_feat[feature], attention_feat)
                Observed[feature] = Feature_feat[feature].reshape(seq_len, batch_size, num_nodes, hidden_dim).transpose(0, 1)


        return Observed



