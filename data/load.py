'''For the Gene Embedding Data'''
import pickle
pickle_path = "./data/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle."
with open(pickle_path, "rb") as f:
    gene_embedding = pickle.load(f)
# print(type(gene_embedding))
# print(len(gene_embedding))
# first_key = list(gene_embedding.keys())[0]
# print(first_key, type(gene_embedding[first_key]), len(gene_embedding[first_key]))


