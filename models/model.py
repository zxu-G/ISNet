import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from .diffusion_block import DifBlock
from .background_block import BGBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decompose.estimation_gate import EstimationGate
from models.RevIN import RevIN
from models.delay_layer import Delay_aware
from models.encoder import Encoder
import os
from .losses import masked_mae



def re_max_min_normalization(x, _max, _min):
    r"""
    Max-min re-normalization

    _max: float
        Max
    _min: float
        Min
    """
    # x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def reconstructed_loss(gated_history, dif_backcast, dif_backcast_res, bg_backcast, null_val=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(gated_history)
    else:
        mask = (gated_history!=null_val)

    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    dif_mae = torch.abs(gated_history - dif_backcast)
    bg_mae = torch.abs(dif_backcast_res - bg_backcast)

    dif_mae = dif_mae * mask
    bg_mae = bg_mae * mask
    dif_mae = torch.where(torch.isnan(dif_mae), torch.zeros_like(dif_mae), dif_mae)
    bg_mae = torch.where(torch.isnan(bg_mae), torch.zeros_like(bg_mae), bg_mae)

    dif_loss = torch.mean(dif_mae)
    bg_loss = torch.mean(bg_mae)

    return dif_loss, bg_loss


class DecomposeLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.feature_num = model_args['feature_num']
        self.estimation_gate = EstimationGate(feature_emb_dim=model_args['feature_emb_dim'],  feature_num=self.feature_num, hidden_dim=64, encoder_dim = model_args['encoder_dim'])
        self.dif_layer      = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.bg_layer      = BGBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.S4_max = model_args['S4_max']
        self.S4_min = model_args['S4_min']
        self.recon_loss = reconstructed_loss

    def forward(self, history_data: torch.Tensor, dynamic_graph: torch.Tensor, dynamic_graph2: torch.Tensor,  feature_args, Prior, Observed, Prior2):

        """decompose layer
        """


        gated_history_data = self.estimation_gate(feature_args, Prior, Observed, history_data)

        dif_backcast_seq_res, dif_forecast_hidden, dif_backcast_seq = self.dif_layer(history_data=history_data, gated_history_data=gated_history_data,
                                                                                     dynamic_graph=dynamic_graph, dynamic_graph2=dynamic_graph2)
        bg_backcast_seq_res, bg_forecast_hidden, bg_backcast_seq = self.bg_layer(dif_backcast_seq_res, Prior, Observed, Prior2)

        dif_loss, bg_loss = self.recon_loss(gated_history_data[:, -dif_backcast_seq.shape[1]:, :, :], dif_backcast_seq, dif_backcast_seq_res, bg_backcast_seq, 0.0)


        return bg_backcast_seq_res, dif_forecast_hidden, bg_forecast_hidden, dif_loss, bg_loss


class ISNet(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self._in_feat       = model_args['num_feat']
        self._hidden_dim    = model_args['num_hidden']          # 32
        self.encoder_dim    = model_args['encoder_dim']         # 96

        self._forecast_dim  = 256
        self._output_hidden = 512
        self.seq_len0 = model_args['seq_length']
        self._num_nodes     = model_args['num_nodes']
        self._k_s           = model_args['k_s']
        self._k_t           = model_args['k_t']
        self._num_layers    = 3
        self.feature_emb_row = [300, 310, 542, 628, 44, 360, 720, 217, 255, 309]
        self.feature_args = model_args['feature_args']
        self.feature_names = model_args['feature_names']
        self.feature_emb_dim = model_args['feature_emb_dim']
        self.encoder = Encoder(input_dim=self.feature_emb_dim * 6, output_dim=self.encoder_dim, ff_dim=64)
        self.dropout = nn.Dropout(model_args['dropout'])


        # RevIn    x_in：（batch_size、sequence_length、num_features）     x_out：（batch_size、prediction_length、num_features）
        self.revin = model_args['revin']   # True
        if self.revin: self.revin_layer = RevIN(num_features=self._num_nodes, affine=True, subtract_last=False)

        model_args['dy_graph']  = True

        self._model_args    = model_args

        # start embedding layer
        # S4 embedding
        self.embedding      = nn.Linear(self._in_feat, self._hidden_dim)


        # feature embedding
        for i, feature in enumerate(self.feature_names):
            if self.feature_args[feature]:
                setattr(self, f"{feature}_emb",
                        nn.Parameter(torch.empty(self.feature_emb_row[i], model_args['feature_emb_dim'])))

        self.reset_parameter()


        # Decomposed Spatial Temporal Layer
        self.layers = nn.ModuleList([DecomposeLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecomposeLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))


        self.delay_layer = Delay_aware(self._hidden_dim, num_heads=4, bias=True, **model_args)

        # dynamic hidden graph constructor
        self.dynamic_graph_constructor  = DynamicGraphConstructor(**model_args)
        

        # output layer
        self.out_fc = nn.Linear(self._forecast_dim, 1)
        self.dif_fc1 = nn.Linear(self._forecast_dim, model_args['gap'])
        self.dif_fc2 = nn.Linear(1, self._forecast_dim)


    def reset_parameter(self):      # todo

        for feature in self.feature_names:
            if self.feature_args[feature]:
                nn.init.xavier_uniform_(getattr(self, f'{feature}_emb'))



    def _prepare_inputs(self, history_data):
        num_feat    = self._model_args['num_feat']           # num_feat=1

        # feature embedding
        dot_num = [_ - 1 for _ in self.feature_emb_row]

        f_feat = 0
        Feature_feat = {}

        # print('history_data.shape', history_data.shape)

        for i, feature in enumerate(self.feature_names):
            if self.feature_args[feature]:
                f_feat += 1
                Feature_feat[feature] = getattr(self, f"{feature}_emb")[(history_data[:, :, :, f_feat] * dot_num[i]).type(torch.LongTensor)]


        history_data = history_data[:, :, :, :num_feat]

        return history_data, Feature_feat



    def forward(self, history_data, real_val):
        """Feed forward of ISNet.
        """

        # RevIn
        if self.revin:
            history_data[:, :, :, 0] = self.revin_layer(history_data[:, :, :, 0], 'norm')


        # ==================== Prepare Input Data ==================== #
        history_data, Feature_feat = self._prepare_inputs(history_data)
        _, Feature_feat2 = self._prepare_inputs(real_val)            # Feature_feat2: priori feature of the future

        history_data   = self.embedding(history_data)
        # print('history_data.shape', history_data.shape)


        Observed = self.delay_layer(history_data, Feature_feat)

        ## Encoder
        Prior = self.encoder(Feature_feat)
        Prior2 = self.encoder(Feature_feat2)


        # ========================= Construct Graphs ========================== #
        dynamic_graph = self.dynamic_graph_constructor(history_data=history_data, Prior=Prior, Observed = Observed, use_X=True)
        dynamic_graph2 = self.dynamic_graph_constructor(Prior=Prior, use_X=False)


        dif_forecast_hidden_list = []
        bg_forecast_hidden_list = []
        dif_loss_sum = 0
        bg_loss_sum = 0

        bg_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):

            _, seq_len, _, _ = bg_backcast_seq_res.shape
            dynamic_graphs = []
            idx = ((self.seq_len0 - seq_len) / 2) + 1

            for _graph in dynamic_graph:
                dynamic_graphs.append(_graph[:, int(0 + idx):int(0 - idx)])

            dynamic_graphs2 = []
            for _graph in dynamic_graph2:
                dynamic_graphs2.append(_graph[:, int(0 + idx):int(0 - idx)])

            feature_args = self.feature_args
            bg_backcast_seq_res, dif_forecast_hidden, bg_forecast_hidden, dif_loss, bg_loss = layer(
                bg_backcast_seq_res, dynamic_graphs, dynamic_graphs2, feature_args, Prior, Observed, Prior2)

            dif_forecast_hidden_list.append(dif_forecast_hidden)
            bg_forecast_hidden_list.append(bg_forecast_hidden)
            dif_loss_sum += dif_loss
            bg_loss_sum += bg_loss

        # Output Layer
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        dif_forecast_hidden = self.dif_fc1(F.relu(dif_forecast_hidden))
        dif_forecast_hidden = dif_forecast_hidden.transpose(1,2).contiguous().view(dif_forecast_hidden.shape[0], dif_forecast_hidden.shape[2], -1)
        dif_forecast_hidden = dif_forecast_hidden.transpose(1, 2).unsqueeze(-1)
        dif_forecast_hidden = self.dropout( self.dif_fc2(dif_forecast_hidden) )

        bg_forecast_hidden = sum(bg_forecast_hidden_list)
        forecast_hidden     = dif_forecast_hidden + bg_forecast_hidden
        
        # regression layer
        forecast = self.out_fc(forecast_hidden)
        forecast = forecast.squeeze(-1).transpose(1, 2).contiguous()


        #  RevIn
        if self.revin:
            forecast = forecast.transpose(1, 2)
            forecast = self.revin_layer(forecast, 'denorm')
            forecast = forecast.transpose(1, 2)

        return forecast, dif_loss_sum, bg_loss_sum
