import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import IPythonConsole

import numpy as np
import pandas as pd
from IPython.display import SVG

from preproc_multilabel import multilabelizer
from mol_train_test import mol_train_test, errors_per_class
from sklearn.metrics import hamming_loss, average_precision_score

"""TRAINING DATA AND TESTING DATA PREPROCESSING"""
# IMPORT DATABASE AND LABELS
dataset_raw = pd.read_csv(r'database_raw_fixing.csv')
labels_raw, counts_original = multilabelizer()

dataset_inchi, labels, all_mols, y_train, y_test,\
        train_fps_array, test_np_fps_array = mol_train_test(dataset_raw, labels_raw, test_size=0.001, nbits = 1024, random_state=3042)

"""NEURAL NETWORK"""
#The neural network model
model = Sequential([
    Dense(1024, input_shape=(1024,), activation= "relu"),
    Dense(256, activation= "sigmoid"),
    Dense(128, activation= "sigmoid"),
    Dense(64, activation= "sigmoid"),    
    BatchNormalization(axis=1),
    Dense(20, activation= "sigmoid")
])
model.summary()

#Compiling the model
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=["accuracy"])

early_stop = EarlyStopping(monitor='val_loss', patience = 10)

#Training the model
history = model.fit(np.array(train_fps_array), y_train.to_numpy(), validation_split=0.1,\
          batch_size=512, epochs= 30, shuffle=True, verbose=1, callbacks=[early_stop])

#Predictions with test dataset
predictions = model.predict(np.array(test_np_fps_array), batch_size=1, verbose=1)

""" Creating results dataframe """

# CREATING RESULTS AND CORRECT LABELS DF
results = pd.DataFrame(predictions, columns = labels.columns)
results_bin = results.round(0).astype(int)

correct_labels = y_test.reset_index(drop=True)

counts_resuls = results_bin.apply(pd.value_counts)
counts_correct = correct_labels.apply(pd.value_counts)

ham_loss = hamming_loss(correct_labels, results_bin)
APS = average_precision_score(correct_labels, results_bin, average='weighted')

print('Hamming loss = {:.4f}  | Average Precision Score = {:.4f}'.format(ham_loss, APS))
"""###################################################################"""
# CALCULATING ERRORS PER CLASS

errors_class, error_counts = errors_per_class(results_bin, correct_labels) 
results_inchi = pd.concat([dataset_inchi[['SMILES', 'CAS-RN']].loc[y_test.index].reset_index(), results_bin], axis=1)
results_inchi['ans alc'] = errors_class['ketone']
"""###################################################################"""

# TRYING KERAS VISUALIZATION
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png')

# Plot training & validation accuracy values
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
""" END """

"""###################################################################"""
database_raw_fixing = dataset_inchi

error_inchi = []

for index in range(database_raw_fixing.shape[0]):
    if database_raw_fixing.Formula[index] not in database_raw_fixing.InChI[index]:
        error_inchi.append(index)
        
# Checking label errors
errors_check = results_inchi[results_inchi['ans alc'] == 'Wrong']

# Trying to check molecules with 2 errors first
molecules_two_errors = errors_class.isin(['Wrong']).apply(pd.Series.value_counts, axis=1) # Counting the number of Wrong results for each molecule
molecules_two_errors.columns = ['NRights', 'NWrong']   # Changing column names (cannot use 'True')
results_inchi['ans alc'] = molecules_two_errors['NWrong']


# =============================================================================
# database_raw_fixing.to_csv(r'database_raw_fixing.csv', index=False)
# =============================================================================
# =============================================================================
# database_raw_fixing = database_raw_fixing.drop([2, 3, 4, 5, 6, 65, 448, 7009, 7018, 9874, 9961, 10909, 10955, 12716])
# database_raw_fixing.reset_index(inplace=True)
# 
# database_raw_fixing.drop(['index'], axis=1, inplace=True)
# database_raw_fixing_deu = database_raw_fixing[database_raw_fixing.Formula.str.contains('D') == False]
# =============================================================================

model.save('mol_classifier.h5')