from rdkit import Chem, DataStructs
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit.Chem import rdMolDescriptors
import pandas as pd

def mol_train_test(dataset, labels, test_size=0.1, random_state=2019, nbits = 1024):

    # TAKING WRONG INCHIS
    all_mols = [Chem.MolFromSmiles(SMILES_string) for SMILES_string in dataset['SMILES']]
    drop_index = [i for i, mol in enumerate(all_mols) if mol == None]               # FINDING WRONG INCHIS
    
    # DROP FROM MOLS, lABELS, AND DATASET    
    if len(drop_index) != 0:
        labels = labels.drop(drop_index).reset_index(drop=True)
        dataset = dataset.drop(drop_index).reset_index(drop=True)
        
    all_mols = [Chem.MolFromSmiles(SMILES_string) for SMILES_string in dataset['SMILES']]  ### FIND BETTER WAY TO NOT CALCULATE AGAIN!!!!
    
    # TRAIN-TEST SPLITS
    train_mols, test_mols, y_train, y_test = train_test_split(all_mols, labels, test_size=test_size\
                                                              , random_state=random_state)
    
    # CONVERT TRAINING MOLECULES INTO FINGERPRINT AS 256BITS VECTORS
    bi = {}
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=nbits) \
           for m in train_mols]
    
    # PUT ALL EACH OF THE CORRESPONDING 256BITS FINGERPRINTS INTO A LIST
    train_fps_array = []
    for fp in fps:
      arr = np.zeros((1,), dtype= int)
      DataStructs.ConvertToNumpyArray(fp, arr)
      train_fps_array.append(arr)
    
    # CONVERT InChi STRINGS INTO MOLECULES FOR TEST DATA
    test_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(test_m, radius=2, bitInfo= bi, nBits=nbits) \
            for test_m in test_mols]
    
    #Convert testing fingerprints into binary, and put all testing binaries into arrays
    test_np_fps_array = []
    for test_fp in test_fps:
      test_arr = np.zeros((1,), dtype= int)
      DataStructs.ConvertToNumpyArray(test_fp, test_arr)
      test_np_fps_array.append(test_arr)
     
    return dataset, labels, all_mols, y_train, y_test, train_fps_array, test_np_fps_array

def errors_per_class(results_bin, correct_labels):
    
    errors_per_class = results_bin.copy()
    for cl in results_bin.columns:
        new_col = []
        for i in range(results_bin.shape[0]):
            if results_bin[cl][i]==correct_labels[cl][i]:
                new_col.append('Right')
            else:
                new_col.append('Wrong')        
        errors_per_class[cl] = new_col    
    
    error_counts = errors_per_class.apply(pd.value_counts).fillna(int(0))
    return errors_per_class, error_counts
