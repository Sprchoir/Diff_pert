'''For the Adamson Dataset'''
import scanpy as sc
# from GEARS.gears.pertdata import PertData  

# data_dir = "./data"
# dataloader = PertData(data_dir)
# dataloader.load(data_name="adamson")
# adata = dataloader.adata

adata = sc.read_h5ad("./data/perturb_processed.h5ad")
print("Number of Cells:", adata.n_obs)
print("number of Genes:", adata.n_vars)
print("Example of Perturbation:", adata.obs['condition'].unique())