import cairosvg

import pandas as pd
import numpy as np

from keras.models import load_model
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
 
# Options for drawing
DrawingOptions.atomLabelFontSize = 150
DrawingOptions.dotsPerAngstrom = 300
DrawingOptions.bondLineWidth = 2.5

# Load model and labels
model = load_model('mol_classifier.h5')
labels = np.array(['alcohol', 'aldehyde', 'alicycle', 'amide', 'aromatic', 'carbocycle',
       'carboxylic acid', 'chiral', 'ester', 'ether', 'fused rings', 'ketone',
       'lactame', 'metal-organic', 'nitrogen heterocycle',
       'oxygen heterocycle', 'sulfide', 'sulfur heterocycle', 'thiol', 'urea'])

# read database
dataset_raw = pd.read_csv(r'database_raw_fixing.csv')
string_molecule = dataset_raw['SMILES'][np.random.randint(1000)]

# read molecule input
string_molecule = 'N#CCCN1CCC(O)(c2ccccc2)CC1'
molecule = Chem.MolFromSmiles(string_molecule)

# Convert input molecule to descriptors
bi = {}
morganFP = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=2, bitInfo= bi, nBits=1024)

train_fps_array = []
morganFP_array = np.zeros((1,), dtype= int)
DataStructs.ConvertToNumpyArray(morganFP, morganFP_array)
train_fps_array.append(morganFP_array)

# Classification
prediction = model.predict(np.array(train_fps_array), batch_size=1, verbose=1)

result = pd.DataFrame(prediction, columns = labels)
result_bin = result.round(0).astype(int)
result_labels = result_bin.astype(bool).to_numpy().tolist()

result_confidence = result.to_numpy()

# Display Result
result_display = labels[tuple(result_labels)]
result_confidence = [conf if round(conf) == 1.0 else 1-conf for conf in result_confidence[0]]

# Draw Image
Draw.MolToFile( molecule, "temp.svg" )
cairosvg.svg2png( url='./temp.svg', write_to= "molecule.png" )