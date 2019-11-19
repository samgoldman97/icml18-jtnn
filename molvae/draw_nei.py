import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser

import numpy as np
from jtnn import *
import sys

parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

#z0 = [
#    "CN1C(C2=CC(NC3C[C@H](C)C[C@@H](C)C3)=CN=C2)=NN=C1",
#    "COC1=CC(OC)=CC([C@@H]2C[NH+](CCC(F)(F)F)CC2)=C1",
#    "COC1=CC(OC)=CC([C@@H]2C[NH+](CCC(F)(F)F)CC2)=C1"
#]

names, smiles = [], []
with open('cayman_mol_samples_hidden350.txt') as f:
    #f.readline()
    for line in f:
        if line.rstrip() == 'None':
            continue
        fields = line.rstrip().split()
        names.append(fields[0])
        smiles.append(fields[0])

batch_size = 10000

for i in range(math.ceil(len(smiles) // batch_size) + 1):
    smiles_batch = smiles[i*batch_size:(i+1)*batch_size]
    names_batch = smiles[i*batch_size:(i+1)*batch_size]

    z0 = model.encode_latent_mean(smiles_batch).squeeze()
    z0 = z0.data.cpu().numpy()

    for smile_idx, name in enumerate(names_batch):
        print('>{}'.format(name))
        print('\t'.join([ str(field) for field in z0[smile_idx] ]))
        sys.stdout.flush()
