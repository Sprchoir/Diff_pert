import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        # MLP block with residual connection.
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

        nn.init.kaiming_uniform_(self.fc[1].weight, nonlinearity="relu")
        if isinstance(self.shortcut, nn.Linear):
            nn.init.kaiming_uniform_(self.shortcut.weight, nonlinearity="linear")

    def forward(self, x):
        return self.fc(x) + self.shortcut(x)

class DecoderNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        dropout = configs["decoder"]["dropout"]
        gene_dim = configs["data"]["num_HVG"]
        num_pc = configs["data"]["num_pc"]
        hidden_dims = configs["decoder"]["hidden_dims"]

        # MLP blocks
        layers = []
        in_dim = num_pc
        for h_dim in hidden_dims:
            layers.append(DecoderBlock(in_dim, h_dim, dropout=dropout))
            in_dim = h_dim
        self.blocks = nn.ModuleList(layers)

        # Final output layer
        self.head = nn.Linear(in_dim, gene_dim)
        nn.init.kaiming_uniform_(self.head.weight, nonlinearity="linear")

        # Non-negative output activation
        self.out_act = nn.Softplus(beta=1.0)

    def forward(self, Y_low):
        h = Y_low
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)           # (B, G)
        out = self.out_act(out)      # >= 0
        return out