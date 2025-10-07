import torch
from torch import nn

class MLPBlock(nn.Module):
    def __init__(self, dropout, in_dim, out_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim + x_dim),
            nn.Linear(in_dim + x_dim, out_dim),
            nn.SiLU(),
            # nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, input, cond_emb):
        input_all = torch.cat([input, cond_emb], dim=-1)
        # Residual Connection
        return self.net(input_all) + self.shortcut(input)
    
class ConditionMLPBlock(nn.Module):
    def __init__(self, x_dim, embedding_dim, time_emb_dim):
        super().__init__()
        cond_dim = x_dim + embedding_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.SiLU(),
            nn.Linear(512, x_dim),
            nn.LayerNorm(x_dim),
            nn.SiLU()
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.LayerNorm(time_emb_dim),
            nn.SiLU()
        )

    def forward(self, x, time, embedding):
        # Concatenate condition
        time_emb = self.time_mlp(time)
        cond = torch.cat([x, time_emb, embedding], dim=-1)
        return self.net(cond)

class Genet(nn.Module):
    """
    MLP with residual mechanism for diffusion model
    Inputs:
        - noised_y: noised perturbed gene expression (x_t)
        - x: top principal components of baseline gene expression
        - embeddings: perturbation embeddings
        - timesteps: current diffusion timesteps
    Outputs:
        - predicted noise   
    """
    def __init__(self, configs):
        super(Genet, self).__init__()
        if configs["data"]["pca"]:
          x_dim = configs["data"]["num_pc"]
        else:
          x_dim = configs["data"]["num_HVG"]
        embedding_dim = configs["model"]["embedding_dim"]
        hidden_dims = configs["model"].get("hidden_dims", [512, 512, 256])
        time_emb_dim = configs["model"].get("time_emb_dim", 128)
        dropout = configs["model"]["dropout"]
        
        # Condition MLP
        self.condition_mlp = ConditionMLPBlock(x_dim, embedding_dim, time_emb_dim)

        # Main MLP Structure
        self.mlp_blocks = nn.ModuleList()
        dim = x_dim
        for i in range(len(hidden_dims)):
            self.mlp_blocks.append(
                MLPBlock(dropout, dim, hidden_dims[i], x_dim)
            )
            dim = hidden_dims[i]

        # Output layer
        self.out_layer = nn.Linear(hidden_dims[-1], x_dim)

    def forward(self, noised_y, x, embeddings, timesteps):
        t = timesteps.float().unsqueeze(-1)
        cond_emb = self.condition_mlp(x, t, embeddings)
        h = noised_y
        for block in self.mlp_blocks:
            h = block(h, cond_emb)
        out = self.out_layer(h)
        return out
    
if __name__ == '__main__':
    pass