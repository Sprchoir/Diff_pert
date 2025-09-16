# DiffPert

## Model

1. Currently, a basic DDPM/DDIM model is used, with modules such as the scheduler implemented via the "diffusers" package. The number of diffusion steps can be set in the configs.yaml. Other approaches, such as Score-based diffusion or EDM, can also be considered. ./src/models/diffusion.py

2.	The neural network is somehow referred to Squidiff, which is an MLP that incorporates a conditional residual mechanism. At least based on my empirical test result, the model performs better with the residual mechanism. ./src/models/net.py

3.  The Trainer class is designed to manage the training and sampling process for a PyTorch-based model. ./src/training/solver.py

## Results

Not many results till now.

On the whole Adamson dataset, training loss (to predict the noise) converges to 0.35 after about 250 epoches with validation loss (to predict the noise) around 0.4. That's not good so the test loss of the whole diffusion process cumulates to a high level.

I think the neural network and the data preprocessing both need to be improved. On the other hand, the size of dataset may influence the performance since there are only 86 kinds of gene perturbations in the adamson, though over 30000 pieces of data.

And I'll construct a decoder for the inversion of PCA later.

