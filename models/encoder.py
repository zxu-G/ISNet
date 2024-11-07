import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):

    def __init__(
        self,
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


    def forward(self, Feature_feat):

        x = torch.cat([Feature_feat['ELE'], Feature_feat['Bt'], Feature_feat['Lat'], Feature_feat['Lon'],
                                       Feature_feat['Solar_ELE'], Feature_feat['Solar_AZI'] ], dim=-1)
        x_proj = self.projection(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x_proj + x
        x = self.ln(x)


        return x



