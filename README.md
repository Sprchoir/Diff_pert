# Diff_Perturbation

## Model

This diffusion model generates Y conditioned on X and Emb.
--- X denotes the unperturbed gene expression
--- Y denotes the perturbed gene expression
--- Emb denotes the gene embeddings from GenePT which is the perturbation label

1. Data preprocessing: scanpy package is used to implement the normalization, log transformation, highly variable genes filtering and PCA, kinda referred to Squidiff. And for the PCA preprocessing, training/validation/test dataset is processed independently, to better deal with the potentially different or even unseen cell types/perturbations in real application. So the PCA parameters are unknown for test and a decoder will be used to reconstruct full gene expression.

2. Diffusion model: a basic DDPM/DDIM model is used currently, partly implemented via the "diffusers" package. The number of diffusion steps can be set in the configuration yaml file. DDPM with 1000 timesteps is somehow slow. Other diffusion models, such as Score-based diffusion or EDM, can also be considered.

3. The denoising neural network is an MLP with residual connection, predicting the added gaussian noise. The conditioning variable (timesteps, Emb, X) is first encoded to the same dimension with the current noised data and then mapped to the target random noise together. The network is trained using maximum mean discrepancy loss (refered to STATE). Transformer can also be applied if needed.

4. The decoder maps the top PC of Y back to Y_HVG (1000 of 5060 in the adamson dataset). A semi-supervised decoder is used, like the training data is partly from the real pc of Y and partly from the generation of our trained diffusion model.

## Results

I've constrcuted the whole pipeline of the model on the whole Adamson dataset and adjusted some model settings.

Here is the model performance for the whole model:
The overall MSE is about 0.1 and MSE for top 20 DE is about 0.55, Pearson R^2 is about 0.07.
The low-rank diffusion MSE is about 0.05.



