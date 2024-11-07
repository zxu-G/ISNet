import torch
import torch.nn as nn
import numpy as np

class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, **model_args):
        super().__init__()
        # gated temporal conv
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        # graph conv
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.num_matric = 2 * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric, self.hidden_dim)

        # others
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):

        out = [X_0]
        for graph in support:
            if len(graph.shape) == 3:
                graph = graph.unsqueeze(1)

            H_k = torch.matmul(graph, X_k)
            out.append(H_k)

        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)

        return out


    def forward(self, X, dynamic_graph):

        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        support = []

        # dynamic graph
        if self.use_dynamic_hidden_graph:
            # k_order is caled in dynamic_graph_constructor component
            support = support + dynamic_graph

        # parallelize
        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        # X: batch_size, seq_len, num_nodes, kernel_size * hidden_dim
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        # X_k: batch_size, seq_len, kernel_size * num_nodes, hidden_dim
        X_k = out.transpose(-3, -2).reshape(batch_size,
                                            seq_len, kernel_size * num_nodes, num_feat)
        # Nx3N 3NxD -> NxD: batch_size, seq_len, num_nodes, hidden_dim
        hidden = self.gconv(support, X_k, X_0)


        return hidden
