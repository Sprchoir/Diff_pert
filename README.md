# DiffPert

## Model

1. Currently, a basic DDPM/DDIM model is used, with modules such as the scheduler implemented via the "diffusers" package. The number of diffusion steps can be set in the configs.yaml. Other approaches, such as Score-based diffusion or EDM, can also be considered. ./src/models/diffusion.py

2.	The neural network is somehow referred to Squidiff, which is an MLP that incorporates a conditional residual mechanism. ./src/models/net.py

3.  The Trainer class is designed to manage the training and sampling process for a PyTorch-based model. ./src/training/solver.py

## Results

Not many results till now.

I'll update later.

