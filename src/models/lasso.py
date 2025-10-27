import torch
import torch.nn as nn

class LassoRefiner(nn.Module):
    def __init__(self, configs):
        """
        A learnable linear layer with L1 penalty for output refinement.
        """
        super().__init__()
        gene_dim = configs["data"]["num_HVG"]
        self.linear = nn.Linear(gene_dim, gene_dim, bias=True)
        self.lambda_l1 = float(configs["decoder"].get("l1_lambda", 1e-2))
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def loss_fn(self, refined_output, target_output):
        mse_loss = nn.functional.mse_loss(refined_output, target_output)
        l1_penalty = torch.mean(torch.abs(self.linear.weight))
        return mse_loss + self.lambda_l1 * l1_penalty