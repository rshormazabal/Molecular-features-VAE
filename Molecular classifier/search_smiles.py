from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import numpy as np

Draw.DrawingOptions.includeAtomNumbers = False

def substructure(mol, structure, structure_name):
    patt = Chem.MolFromSmarts(structure)
    matches = mol.GetSubstructMatches(patt)
    if len(matches) != 0:
        print(structure_name)
        return True
    else:
        return False

if __name__ == "__main__":

    # IMPORT WITH PANDAS
    raw_dataset = pd.read_csv(r'new_database_checked_nocharge.csv', )
    labels = raw_dataset['Classification'].str.split(pat='; ')
    dataset = raw_dataset.copy()

    # CONVERT InChi STRINGS INTO MOLECULES FOR TRAINING DATA
    dataset_mols = [Chem.inchi.MolFromInchi(inChi_string) for inChi_string in dataset['InChI']]

    results = []
    wrong = []
    for i, mol in enumerate(dataset_mols):
        alc = substructure(mol,'[NX3][CX3](=[OX1])[#6]', 'Amide index={}'.format(i)) 
        if alc == True:
            results.append('amide')
        if alc == False:
            results.append('')

raw_dataset['amide'] = np.where(raw_dataset['Classification'].str.contains('amide'), 'amide', 'no')
raw_dataset['results'] = results
raw_dataset['Check amide'] = np.where((raw_dataset['amide']=='no') & (raw_dataset['results']=='amide') | \
           (raw_dataset['amide']=='amide') & (raw_dataset['results']==''), 'Check', 'Dont check')

raw_dataset.rename({'Check': 'Check Alcohol'}, inplace=True)
raw_dataset.drop('alc', axis=1, inplace=True)
counts = raw_dataset['Check amide'].value_counts()

# =============================================================================
#     for index, mol in enumerate(dataset_mols):
#         results.append([])
#         ald = substructure(mol,'[c,C,#1][CX3H1](=O)', 'Aldehyde index={}'.format(index))
#         alc = substructure(mol,'[#6][OX2H]', 'Alcohol index={}'.format(index)) 
#         ket = substructure(mol,'[#6][CX3](=O)[#6]', 'Ketone index={}'.format(index))      
#         ester = substructure(mol,'[CX3](=O)[OX2H0][#6]', 'Ester index={}'.format(index))  
#         ether = substructure(mol,'[OD2]([#6])[#6]', 'Ether index={}'.format(index))           
#         acid = substructure(mol,'[CX3](=O)[OX2H1]', 'Acid index={}'.format(index))   
#         
#         if ald== True:
#             results[index].append('Aldehydes')
#         if alc == True:
#             results[index].append('Alcohols')
#         if ket == True:
#             results[index].append('Ketones')
#         if ester == True:
#             results[index].append('Esters')
#         if ester == False and ether == True:
#             results[index].append('Ethers')      
#         if acid == True:
#             results[index].append('Acids')
#             
# =============================================================================
dataset.drop(6606, inplace=True)
raw_dataset.reset_index(inplace=True)