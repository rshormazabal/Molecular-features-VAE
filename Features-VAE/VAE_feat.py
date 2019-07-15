import numpy as np
import json
import cairosvg
from rdkit import Chem
from rdkit.Chem import Draw
from model import MoleculeVAE
from rdkit.Chem.Draw import DrawingOptions
from utils import encode_smiles, decode_latent_molecule, interpolate, get_unique_mols


# number of dimensions to represent the molecules
# as the model was trained with this number, any operation made with the model must share the dimensions.
# trained_model 0.99 validation accuracy
# trained with 80% of ALL CHEMBL molecules, validated on the other 20.

latent_dim = 292
trained_model = '/Users/Rodrigo/untitled/Features-VAE/chembl_23_model.h5'
charset_file = '/Users/Rodrigo/untitled/Features-VAE/charset.json'
aspirin_smiles = 'CC(=O)Oc1ccccc1C(=O)O'

# Load charset and model
with open(charset_file, 'r') as outfile:
    charset = json.load(outfile)

model = MoleculeVAE()
model.load(charset, trained_model, latent_rep_size=latent_dim)

aspirin_latent = encode_smiles(aspirin_smiles, model, charset)

reconstructed_aspirin = decode_latent_molecule(aspirin_latent, model, charset, latent_dim)
original = Chem.MolFromSmiles(aspirin_smiles)
reconstructed = Chem.MolFromSmiles(reconstructed_aspirin)

# Options for drawing
DrawingOptions.atomLabelFontSize = 150
DrawingOptions.dotsPerAngstrom = 300
DrawingOptions.bondLineWidth = 2.5

Draw.MolToFile(reconstructed, "temp.svg", size=(200, 200))
cairosvg.svg2png(url='./temp.svg', write_to="similar_mol.png")