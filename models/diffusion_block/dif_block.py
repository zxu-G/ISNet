import torch.nn as nn

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decompose.residual_decomp import ResidualDecomp


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None,  **model_args):
        """Diffusion block
        """

        super().__init__()

        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=None, use_pre=use_pre, dy_graph=dy_graph, **model_args)

        # forecast
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        # backcast
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        # esidual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, dynamic_graph2):
        """Diffusion block, containing the diffusion model, forecast branch, backcast branch, and the residual decomposition link.
        """

        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph)

        dynamic_graphs = []
        for _graph in dynamic_graph2:
            dynamic_graphs.append(_graph[:, -1, :, :])
        dynamic_graph = dynamic_graphs

        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph)

        backcast_seq = self.backcast_branch(hidden_states_dif)

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]

        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)




        return backcast_seq_res, forecast_hidden, backcast_seq
