"""Embed molecules using pretrained JTVAE encoder 

Note: To run this,c hange into this directory and use a gpu because it is
hardcoded into some parts of the model 

"""
# Hack to get path 
import sys
import os

from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import torch
import torch.nn as nn

import math, random

import argparse
from fast_jtnn import *
import rdkit
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem
import pickle

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--smiles-list', required=True, type=str, 
                    help="Path to file containing smiles strings")
parser.add_argument('--out', default="embeddings", type=str, 
                    help="Prefix outpath")

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
model.load_state_dict(torch.load(args.model))

if args.gpu: 
    model = model.cuda()

# Map smiles to embedding
embeddings = {}

input_smiles = open(args.smiles_list, "r").readlines()

# Filter title and remove new lines
input_smiles = [j.strip() for index, j in enumerate(input_smiles) if index >= 1]

# Create charge parents
input_mols = [Chem.MolFromSmiles(i) for i in input_smiles]
standardized_mols = [rdMolStandardize.ChargeParent(i) for i in input_mols]
input_smiles = [Chem.MolToSmiles(i) for i in standardized_mols]

out_tensors = model.encode_from_smiles(input_smiles)
output = out_tensors.cpu().detach().numpy()
feature_mapping = {}

pickle_obj = dict(zip(input_smiles, output))
pickle.dump(pickle_obj, open(f"{args.out}.p", "wb"))

loaded_pickle = pickle.load(open(f"{args.out}.p", "rb"))
import pdb
pdb.set_trace()

print(loaded_pickle)
