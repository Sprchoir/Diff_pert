import torch
from torch import nn

class ConditionalResidualBlock(nn.Module):
    def __init__(self, dim, out_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
            #nn.Dropout(0.1)
        )
        self.res_proj = nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()

    def forward(self, input, cond):
        input_cond = torch.cat([input, cond], dim=-1)
        out = self.net(input_cond)
        return out + self.res_proj(input)