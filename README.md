# Diff_Perturbation

## Model

This diffusion model generates Y conditioned on X and Emb.
--- X denotes the unperturbed gene expression
--- Y denotes the perturbed gene expression
--- Emb denotes the gene embeddings from GenePT which is the perturbation label

1. Data preprocessing: scanpy package is used to implement the normalization, log transformation, highly variable genes filtering and PCA, kinda referred to Squidiff. And for the PCA preprocessing, training/validation/test dataset is processed independently, to better deal with the potentially different or even unseen cell types/perturbations in real application. So the PCA parameters are unknown for test and a decoder will be used to reconstruct full gene expression.

2. Diffusion model: a basic DDPM/DDIM model is used currently, partly implemented via the "diffusers" package. The number of diffusion steps can be set in the configuration yaml file. DDPM with 1000 timesteps is somehow slow. Other diffusion models, such as Score-based diffusion or EDM, can also be considered.

3. The denoising neural network is an MLP with residual connection, predicting the added gaussian noise. The conditioning variable (timesteps, Emb, X) is first encoded to the same dimension with the current noised data and then mapped to the target random noise together. The network is trained using maximum mean discrepancy loss (refered to STATE). Transformer can also be applied if needed.

4. The decoder maps the top PC of Y back to Y with full dimensions(5060 in the adamson dataset). I plan to train a semi-supervised decoder, like the training data is partly from the real pc of Y and partly from the generation of our trained diffusion model.

## Results

I've trained the model on the whole Adamson dataset and adjusted some model settings.

Here is the model performance for the diffusion model only (without reconstructing the data):
With PCA of Y and X, training loss and the validation loss (to predict the noise) both converges to less than 0.026 in 200 epoches. And the test loss of the whole diffusion process with 100 diffusion steps cumulates to about 0.15. 
While without PCA, training loss converges to 0.06 and validation loss converges to around 0.11 in 200 epoches. The test loss turns out to be more than 0.6. And it's possible sometimes to only get all NAN output due to too many zeros in the data.

Till now, at least the low-rank diffusion is better...
I think the neural network and the data preprocessing both can be improved. And I don't know whether some steps are correct for genomcis data. Additionally, the size of dataset may limit the performance since there are only 86 kinds of gene perturbations in the adamson, though over 30000 pieces of data.

Next, I'll construct a decoder for the inversion of PCA.

