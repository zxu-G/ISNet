import torch
import torch.nn as nn


class EstimationGate(nn.Module):
    """The estimation gate module."""

    def __init__(self, feature_emb_dim, feature_num, hidden_dim, encoder_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(feature_emb_dim * 5 + encoder_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature_args, Prior, Observed, history_data):

        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two signals."""

        batch_size, seq_length, _, _ = Observed['SYMH'].shape

        Observed = torch.cat([v for k, v in Observed.items()], dim=-1)
        estimation_gate_feat = torch.cat([Prior[:, -history_data.shape[1]:], Observed[:, -history_data.shape[1]:], history_data], dim=-1)

        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate


        return history_data
