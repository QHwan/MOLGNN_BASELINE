import numpy as np
import tqdm
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import networkx as nx


def one_hot_encoding(x, cand_list):
    if x not in cand_list:
        print("{} is not in {}.".format(x, cand_list))
        exit(1)

    one_hot_vec = np.zeros(len(cand_list))
    one_hot_vec[cand_list.index(x)] = 1
    return list(one_hot_vec)

atom_type_list = ['H', 'He', 'Li', 'Be', 'B',
                'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P',
                'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn',
                'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br',
                'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs',
                'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re',
                'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At',
                'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu', 'Am',
                'Cm', 'Bk', 'Cf', 'Es', 'Fm']
valence_list = [0, 1, 2, 3, 4, 5, 6]
formal_charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
radical_list = [0, 1, 2]
hybridization_list = [Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                    Chem.rdchem.HybridizationType.S,
                    Chem.rdchem.HybridizationType.UNSPECIFIED,]
aromatic_list = [0, 1]
num_h_list = [0, 1, 2, 3, 4]
degree_list = [0, 1, 2, 3, 4, 5]

bond_type_list = [Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC]
conjugate_list = [0, 1]
ring_list = [0, 1]
stereo_list = [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOANY,
               Chem.rdchem.BondStereo.STEREOCIS,
               Chem.rdchem.BondStereo.STEREOTRANS,
               Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ
               ]

def get_atom_feature(atom, explicit_H=False):
    out = one_hot_encoding(atom.GetSymbol(),
                            atom_type_list)
    out += one_hot_encoding(atom.GetHybridization(),
                            hybridization_list)
    if not explicit_H:
        out += one_hot_encoding(atom.GetTotalNumHs(),
                                num_h_list)

    out += one_hot_encoding(atom.GetFormalCharge(), formal_charge_list)
    out += [int(atom.GetIsAromatic())]

    return(out)

def bond_features(bond, explicit_H=False):
    out = one_hot_encoding(bond.GetBondType(),
                            bond_type_list)
    out += [bond.GetIsConjugated()]
    out += [bond.IsInRing()]
    return(out)


def screening_unrealistic(pre_smiles_list, pre_y_list):
    smiles_list, y_list = [], []
    for i, (smiles, y) in enumerate(zip(pre_smiles_list, pre_y_list)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("molecule {} is not defined well".format(smiles))
            continue
        smiles_list.append(smiles)
        y_list.append(y)

    return np.array(smiles_list), np.array(y_list)


def get_mol_fea(smiles_list, label_list):
    x_list = []
    y_list = []
    edge_index_list = []
    node_dim = len(atom_type_list + hybridization_list + num_h_list+formal_charge_list) + 1

    for i, smiles in tqdm.tqdm(enumerate(smiles_list), total=len(smiles_list)):
        
        if smiles[-1] == ' ':
            smiles = smiles[:-1]

        mol = Chem.MolFromSmiles(smiles)

        atom_list = mol.GetAtoms()
        n_node = len(atom_list)

        x = np.zeros((n_node, node_dim))
        y = np.zeros(1)
        edge_index = []

        for j, atom in enumerate(atom_list):
            x[j] += get_atom_feature(atom)

        y[0] += label_list[i]

        for j in range(n_node):
            for k in range(j+1, n_node):
                bond = mol.GetBondBetweenAtoms(j, k)
                if bond is not None:
                    edge_index.append([j, k])
                    edge_index.append([k, j])

        x_list.append(x), y_list.append(y), edge_index_list.append(edge_index)

    return x_list, y_list, edge_index_list