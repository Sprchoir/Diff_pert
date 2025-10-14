# Diff_Perturbation

## Model

This diffusion model generates Y conditioned on X and Emb.
--- X denotes the unperturbed gene expression
--- Y denotes the perturbed gene expression
--- Emb denotes the gene embeddings from GenePT which is the perturbation label

1. Data preprocessing: scanpy package is used to implement the log transformation, highly variable genes filtering and PCA. PCA is done on the training/validation/test dataset independently, to better deal with the potentially different or even unseen cell types/perturbations in real application. So PCA parameters are unknown for test and a decoder will be used to reconstruct full gene expression.

2. Diffusion model: a basic DDPM/DDIM model is used currently, partly implemented via the "diffusers" package. Other diffusion models, such as Score-based diffusion or EDM, can also be considered.

3. The denoising neural network is an MLP with residual connection, predicting the gaussian noise. The network is trained using maximum mean discrepancy loss (refered to STATE).

4. The decoder maps the top PC of Y back to Y. A semi-supervised decoder is used, like the training data partly from the real pc of Y and partly from the generation of trained diffusion model.

## Results

I've constrcuted the whole pipeline of the model on the whole Adamson dataset and adjusted some model settings. 
Cell Representation(how to take advantage of the gene embeddings) and decoder network still have potential to be improved.

Here is the model performance for the whole model:
The overall MSE is about 0.1.
The low-rank diffusion MSE is about 0.05.



