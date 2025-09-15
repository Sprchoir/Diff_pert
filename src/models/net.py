import torch
from torch import nn
from .block import ConditionalResidualBlock

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
        x_dim = configs["data"]["num_pc"]
        embedding_dim = configs["model"]["embedding_dim"]
        embedding_proj_dim = configs["model"].get("embedding_proj_dim", 512)
        hidden_dims = configs["model"].get("hidden_dims", [512, 512, 256])
        time_emb_dim = configs["model"].get("time_emb_dim", 128)
        cond_dim = x_dim + embedding_proj_dim + time_emb_dim   # noised_y + x + embedding

        # timestep embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Embedding projection
        self.embedding_proj = nn.Linear(embedding_dim, embedding_proj_dim)

        # input layer
        self.input_layer = nn.Linear(x_dim, hidden_dims[0])

        # MLP blocks
        self.mlp_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.mlp_blocks.append(
                # conditional residual mechanism
                ConditionalResidualBlock(hidden_dims[i], hidden_dims[i+1], cond_dim)
            )

        # output layer
        self.output_layer = nn.Linear(hidden_dims[-1], x_dim)

    def forward(self, noised_y, x, embeddings, timesteps):
        emb_proj = self.embedding_proj(embeddings)
        t = timesteps.float().unsqueeze(-1)
        t_emb = self.time_mlp(t)

        cond = torch.cat([x, emb_proj, t_emb], dim=-1)
        input = self.input_layer(noised_y)
        for block in self.mlp_blocks:
            output = block(input, cond)
        out = self.output_layer(output)
        return out
    
if __name__ == '__main__':
    pass