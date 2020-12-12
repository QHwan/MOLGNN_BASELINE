import numpy as np
import pandas as pd
import random
import argparse
import drug_fea
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_split', type=int, default=10)
args = parser.parse_args()

if args.dataset.lower() == 'esol':
    data_filename = './esol/esol.csv'
    out_filename = './esol/esol.npz'
    df = pd.read_csv(data_filename)
    pre_smiles_list = df['smiles']
    pre_y_list = df['measured log solubility in mols per litre']

# screen unrealistic molecules in dataset with RDKit
smiles_list, y_list = drug_fea.screening_unrealistic(pre_smiles_list, pre_y_list)

# Drug smiles: molecular graph encoding
# Data list is torch_geometric.data class
x_list, y_list, edge_index_list = drug_fea.get_mol_fea(smiles_list, y_list)

assert (len(smiles_list) == len(y_list)) # Simple Check

# Split list
n_data = len(y_list)
idx_list = list(range(n_data))
np.random.shuffle(idx_list)
split_list = np.array_split(idx_list, args.n_split)

np.savez(out_filename, 
    x=x_list, y=y_list, edge_index=edge_index_list, split=split_list,
    allow_pickles=True)