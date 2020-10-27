# Junction Tree Variational Autoencoder for Molecular Graph Generation

This is a fork of Brian Hie's Python 3 compatible version of the JTVAE as used
in [http://cell.com/cell-systems/fulltext/S2405-4712(20)30364-1](Leveraging
Uncertainty in Machine Learning Accelerates Biological Discovery and Design).
Specifically, the pretrained ZINC embedding models from the old implementation can be used to embed a new set of molecules using the 'molvae/embed.py' script. 

Official implementation of our Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

# Accelerated Version
We have accelerated our code! The new code is in `fast_jtnn/`, and the VAE training script is in `fast_molvae/`. Please refer to `fast_molvae/README.md` for details.

# Requirements
* Linux (We only tested on Ubuntu)
* RDKit (version >= 2017.09)
* Python (version == 2.7)
* PyTorch (version >= 0.2)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Quick Start
The following directories contains the most up-to-date implementations of our model:
* `fast_jtnn/` contains codes for model implementation.
* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.

The following directories provides scripts for the experiments in our original ICML paper:
* `bo/` includes scripts for Bayesian optimization experiments. Please read `bo/README.md` for details.
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training our VAE model.
* `molopt/` includes scripts for jointly training our VAE and property predictors. Please read `molopt/README.md` for details.
* `jtnn/` contains codes for model formulation.


# Contact
Wengong Jin (wengong@csail.mit.edu)
