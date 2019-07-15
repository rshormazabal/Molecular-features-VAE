import cairosvg

import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions

from utils import encode_smiles, decode_latent_molecule, interpolate, get_unique_mols


def pred_(model, string_mol, inputopt):
    # Labels
    labels = np.array(['Alcohol', 'Aldehyde', 'Alicycle', 'Amide', 'Aromatic', 'Carbocycle',
                       'Carboxylic acid', 'Chiral', 'Ester', 'Ether', 'Fused rings', 'Ketone',
                       'Lactame', 'Metal-organic', 'Nitrogen heterocycle',
                       'Oxygen heterocycle', 'Sulfide', 'Sulfur heterocycle', 'Thiol', 'Urea'])

    if inputopt == 'InChI':
        # Read molecule input
        molecule = Chem.inchi.MolFromInchi(string_mol)
    if inputopt == 'SMILES':
        # Read molecule input
        molecule = Chem.MolFromSmiles(string_mol)

    # Convert input molecule to descriptors
    bi = {}
    morganFP = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=2, bitInfo=bi, nBits=1024)

    train_fps_array = []
    morganFP_array = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(morganFP, morganFP_array)
    train_fps_array.append(morganFP_array)

    # Classification
    prediction = model.predict(np.array(train_fps_array), batch_size=1, verbose=1)

    result = pd.DataFrame(prediction, columns=labels)
    result_bin = result.round(0).astype(int)
    result_labels = result_bin.astype(bool).to_numpy().tolist()

    result_confidence = result.to_numpy()

    # Display Result
    result_display = labels[tuple(result_labels)]
    result_confidence_float = [conf for conf in result_confidence[0]]   # use for sorting (list)

    # Transforming into percentages
    result_confidence = [format(n, '.2%') for n in result_confidence_float] # use for stack (str)

    # Formatting results
    result_display = ', '.join(result_display)
    if result_display == '':
        result_display = 'No functional groups found'
    result_confidence = np.column_stack((labels, result_confidence))  # Stack
    result_confidence = result_confidence[np.argsort(result_confidence_float)[::-1]]  # Order descending

    return result_display, result_confidence


def create_similar_mols_(string_mol, model_vae, charset, stdev):

    molecule_latent = encode_smiles(string_mol, model_vae, charset)  # Molecule Latent space

    working_mols = []
    working_mols_smiles = []
    counter = 0
    while len(working_mols) < 9:
        similar_mol_latent = stdev * np.random.randn(1, 292) + molecule_latent  # Perturbation for generation
        similar_mol = decode_latent_molecule(similar_mol_latent, model_vae, charset, 292)
        try:
            mol = Chem.MolFromSmiles(similar_mol)
            if mol and similar_mol not in working_mols and similar_mol != string_mol:
                working_mols_smiles.append(similar_mol)
                working_mols.append(mol)
        except:
            continue
        counter+=1
        print(counter)
    working_mols_inchi = [Chem.inchi.MolToInchi(mol) for mol in working_mols]

    return working_mols_inchi, working_mols_smiles, working_mols


def create_image_(string_mol, inputopt, number=''):
    # Options for drawing
    DrawingOptions.atomLabelFontSize = 150
    DrawingOptions.dotsPerAngstrom = 300
    DrawingOptions.bondLineWidth = 5

    # Read molecule input
    if inputopt == 'InChI':
        molecule = Chem.inchi.MolFromInchi(string_mol)
    if inputopt == 'SMILES':
        molecule = Chem.MolFromSmiles(string_mol)

    Draw.MolToFile(molecule, "temp.svg", size=(800, 800))
    cairosvg.svg2png(url='./temp.svg', write_to="molecule.png"+number)


def molecule_info_(string_mol, inputopt):
    # Read molecule input
    # other_input = if Smiles --> InChI, for information table.
    if inputopt == 'InChI':
        molecule = Chem.inchi.MolFromInchi(string_mol)
        other_input = ['SMILES', Chem.MolToSmiles(molecule)]
    else:
        molecule = Chem.MolFromSmiles(string_mol)
        other_input = ['InChI', Chem.inchi.MolToInchi(molecule)]

    # Different input option
    properties = [other_input[1]]

    # Molecule formula
    properties.append(Chem.rdMolDescriptors.CalcMolFormula(molecule))

    # Molecule mass
    properties.append(Chem.rdMolDescriptors.CalcExactMolWt(molecule))

    # Number of Rings
    properties.append(Chem.rdMolDescriptors.CalcNumRings(molecule))

    # Number of Aromatic Rings
    properties.append(Chem.rdMolDescriptors.CalcNumAromaticRings(molecule))

    # List of properties
    properties_list = [other_input[0], 'Formula', 'Mol. mass[g/mol]', 'Rings', 'Aromatic Rings']
    properties = np.column_stack((properties_list, properties))

    return properties

