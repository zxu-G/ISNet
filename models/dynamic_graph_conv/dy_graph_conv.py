import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph  # 1 order
        mask = torch.eye(graph.shape[2]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):  # e.g., order = 3, k=[2, 3]; order = 2, k=[2]   self.order=k_s
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]





class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # model args
        self.k_s = model_args['k_s']  # spatial order
        self.k_t = model_args['k_t']  # temporal kernel size
        self.hidden_dim = model_args['num_hidden']
        self.feature_emb_dim = model_args['feature_emb_dim']

        self.fc_x_emb = nn.Linear(self.hidden_dim * 5 + model_args['encoder_dim'], 64)
        self.fc_x_emb2 = nn.Linear(model_args['encoder_dim'], 32)
        self.multi_order = MultiOrder(order=self.k_s)

    def transition_matrix(self, adj):
        r"""
        Description:
        -----------
        Calculate the transition matrix `P`
        P = D^{-1}A = A/rowsum(A)

        Parameters:
        -----------
        adj: np.ndarray
            Adjacent matrix A

        Returns:
        -----------
        P:np.matrix
            Renormalized message passing adj in `GCN`.
        """

        rowsum = adj.sum(dim=3).flatten(2) + 1e-9
        d_inv = rowsum.pow(-1).flatten(2)
        d_inv[torch.isinf(d_inv)] = 0.
        d_inv[torch.isnan(d_inv)] = 0.
        diag = torch.eye(d_inv.size(-1)).to(d_inv.device)
        d_mat = torch.einsum('ijk,kl->ijkl', d_inv, diag)
        P = d_mat @ adj

        return P



    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1], k_order_graph.shape[2],
                    k_order_graph.shape[3] * k_order_graph.shape[4])
                st_local_graph.append(k_order_graph)

        return st_local_graph

    def forward(self, **inputs):
        """Dynamic graph learning module.
        """

        if inputs['use_X']:
            Observed = torch.cat([v for k, v in inputs['Observed'].items()], dim=-1)
            nodevec = torch.cat([inputs['history_data'], inputs['Prior'], Observed], dim=-1)
            nodevec = self.fc_x_emb(nodevec)
            nodevec = torch.tanh(nodevec)
        else:
            nodevec = inputs['Prior']
            nodevec = self.fc_x_emb2(nodevec)
            nodevec = torch.tanh(nodevec)

        similarity = F.relu(torch.matmul(nodevec, nodevec.transpose(3, 2)))

        ##  get_laplacian
        D = torch.diag_embed((torch.sum(similarity, dim=-1) + 1e-9) ** (-1 / 2))
        adj_mx = torch.matmul(torch.matmul(D, similarity), D)

        P1 = torch.transpose(self.transition_matrix(adj_mx), -2, -1)
        P2 = torch.transpose(self.transition_matrix(torch.transpose(adj_mx, -2, -1)), -2, -1)
        double_transition = [P1, P2]

        # multi order
        mul_mx = self.multi_order(double_transition)

        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)


        return dynamic_graphs







