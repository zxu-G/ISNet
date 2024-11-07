import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decompose.residual_decomp import ResidualDecomp
from models.RevIN import RevIN






class TimeMixing(nn.Module):

    def __init__(
        self,
        seq_len: int,
        input_channels: int,
        norm_dim: int,
    ):

        super().__init__()
        self.ln = nn.LayerNorm(norm_dim)
        self.dropout = nn.Dropout(0.1)
        self.fut_feature_mix_layer = nn.Conv2d(in_channels=seq_len,
                                               out_channels=seq_len,
                                               kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_res = F.relu(self.dropout(self.fut_feature_mix_layer(x)))
        x = self.ln(x + x_res)

        return x




class FeatureMixing(nn.Module):

    def __init__(
        self,
        seq_len: int,
        input_channels: int,
        output_channels: int,
        input_dim: int,
        output_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):

        super().__init__()
        self.ln = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_proj = self.projection(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        x = x_proj + x
        x = self.ln(x)


        return x






class BGBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):

        """background block
        """
        super().__init__()
        self.num_feat   = hidden_dim
        self.hidden_dim = hidden_dim
        self._num_nodes = model_args['num_nodes']
        self.output_seq_len0 = 12
        self.num_layer = 3

        self.encoder_dim1 = self.hidden_dim * 5 + model_args['encoder_dim']
        self.past_mix_dim = 256
        self.encoder_dim2 = self.encoder_dim1 + model_args['encoder_dim']
        self.forecast_hidden_dim = forecast_hidden_dim
        self.dropout = nn.Dropout(model_args['dropout'])


        ## past feature mixing
        self.feature_mixing1 = FeatureMixing(seq_len = self.output_seq_len0, input_channels = self._num_nodes, output_channels = self._num_nodes,
                                            input_dim = self.encoder_dim1, output_dim = self.encoder_dim1, ff_dim = 128)
        self.Feature_mixing_layers1 = nn.Sequential( *[self.feature_mixing1 for _ in range(1)])


        ## past time mixing
        self.time_mixing1_1 = TimeMixing(seq_len=34, input_channels=self._num_nodes, norm_dim=self.encoder_dim1)
        self.Time_mixing_layers1_1 = nn.Sequential( *[self.time_mixing1_1 for _ in range(self.num_layer)])

        self.time_mixing1_2 = TimeMixing(seq_len=32, input_channels=self._num_nodes, norm_dim=self.encoder_dim1)
        self.Time_mixing_layers1_2 = nn.Sequential( *[self.time_mixing1_2 for _ in range(self.num_layer)])

        self.time_mixing1_3 = TimeMixing(seq_len=30, input_channels=self._num_nodes, norm_dim=self.encoder_dim1)
        self.Time_mixing_layers1_3 = nn.Sequential( *[self.time_mixing1_3 for _ in range(self.num_layer)])


        # temporal projection
        self.temporal_proj_layer1 = nn.Conv2d(in_channels=34,
                                                out_channels=self.output_seq_len0,
                                                kernel_size=(1, 1), bias=True)
        self.temporal_proj_layer2 = nn.Conv2d(in_channels=32,
                                                out_channels=self.output_seq_len0,
                                                kernel_size=(1, 1), bias=True)
        self.temporal_proj_layer3 = nn.Conv2d(in_channels=30,
                                                out_channels=self.output_seq_len0 ,
                                                kernel_size=(1, 1), bias=True)


        # backcast branch
        self.backcast_fc =  nn.Linear(self.encoder_dim1, self.hidden_dim)

        # residual decomposition
        self.residual_decompose   = ResidualDecomp([-1, -1, -1, hidden_dim])


        ## future feature mixing
        self.feature_mixing2 = FeatureMixing(seq_len=self.output_seq_len0, input_channels=self._num_nodes,
                                            output_channels=self._num_nodes,
                                            input_dim=self.encoder_dim2, output_dim=forecast_hidden_dim, ff_dim=128)
        self.Feature_mixing_layers2 = nn.Sequential(
            *[self.feature_mixing2 for _ in range(1)])


        ## future time mixing
        self.time_mixing2 = TimeMixing(seq_len=self.output_seq_len0, input_channels=self._num_nodes,
                                      norm_dim=forecast_hidden_dim)
        self.Time_mixing_layers2 = nn.Sequential(
            *[self.time_mixing2 for _ in range(self.num_layer)])






    def forward(self, hidden_background_signal, Prior, Observed, Prior2):


        [batch_size, seq_len, num_nodes, num_feat]  = hidden_background_signal.shape


        Observed = torch.cat([v for k, v in Observed.items()], dim=-1)
        hidden_states_bg = torch.cat([hidden_background_signal[:, -seq_len:], Observed[:, -seq_len:], Prior[:, -seq_len:]], dim=-1)

        # Feature mixing
        hidden_states_bg = self.Feature_mixing_layers1(hidden_states_bg)

        # Time mixing
        if seq_len==34:
            hidden_states_bg = self.Time_mixing_layers1_1(hidden_states_bg)
        elif seq_len==32:
            hidden_states_bg = self.Time_mixing_layers1_2(hidden_states_bg)
        elif seq_len==30:
            hidden_states_bg = self.Time_mixing_layers1_3(hidden_states_bg)

        # backcast
        backcast_seq = self.backcast_fc(hidden_states_bg)
        backcast_seq_res = self.residual_decompose(hidden_background_signal, backcast_seq)

        # temporal projection
        if seq_len==34:
            hidden_states_bg = self.temporal_proj_layer1(hidden_states_bg)
        elif seq_len==32:
            hidden_states_bg = self.temporal_proj_layer2(hidden_states_bg)
        elif seq_len==30:
            hidden_states_bg = self.temporal_proj_layer3(hidden_states_bg)


        hidden_states_bg = torch.cat([hidden_states_bg, Prior2], dim=-1)
        # Feature mixing
        hidden_states_bg = self.Feature_mixing_layers2(hidden_states_bg)


        # Time mixing
        forecast_hidden = self.Time_mixing_layers2(hidden_states_bg)




        return backcast_seq_res, forecast_hidden, backcast_seq

