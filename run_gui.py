import datetime
import json
import os
import sys
from functools import partial

import cairosvg
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QApplication
from PyQt5.uic import loadUi
from keras.models import load_model
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw

from classifier_GUI import pred_, create_image_, molecule_info_, create_similar_mols_
from model import MoleculeVAE

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# Show CSC1=NC(=CC(=O)N1)Cl
# Aspirin (acetylsalicylic acid) is an aromatic compound containing both a carboxylic acid functional
# group and an ester functional group. 'CC(=O)OC1=CC=CC=C1C(=O)O'
# Molecule with error InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1


class MainPage(QDialog):
    def __init__(self):
        super(MainPage, self).__init__()
        loadUi('main_page_max.ui', self)
        # Attaching a function to a button in the interface
        self.drawButton.clicked.connect(self.drawMolecule)
        self.analyzeButton.clicked.connect(self.prediction_class)
        self.sendRightButton.clicked.connect(self.createFile)
        self.simButton.clicked.connect(self.similar_mols)
        self.wpCheckBox.stateChanged.connect(self.activate_fixLabels)
        self.inputOptions()

        # Model and input type
        self.model = load_model('mol_classifier.h5')
        self.model_vae = MoleculeVAE()
        self.inputopt = 'SMILES'

        # Results for Classification
        self.res = []
        self.conf = []

        # Options for similar molecules and VAE options
        self.similar_mols = []
        self.sim_mols_smiles = [''] * 9
        self.sim_mols_inchi = [''] * 9
        self.stdev = 0.05
        with open('Features-VAE/charset.json', 'r') as outfile:  # Creating list of caracters
            self.charset = json.load(outfile)

        self.model_vae.load(self.charset, 'Features-VAE/chembl_23_model.h5', latent_rep_size=292)  # Loading model

        # Combo box for type of input
        self.inputOpt.currentIndexChanged.connect(self.chooseInput)

        # Confidence table
        self.confTable.setRowCount(20)

        # Molecule info Table
        self.infoTable.setRowCount(5)

    def drawMolecule(self):
        # Collecting text from a inputString
        molecule = self.inputString.toPlainText()

        # Creating molecule, fetch image and data
        create_image_(molecule, self.inputopt)
        molecule_info_(molecule, self.inputopt)
        self.drawMol.setPixmap(QPixmap('molecule.png'))

        # Fill table for molecular info
        self.infoTable.setEnabled(True)
        properties = molecule_info_(self.inputString.toPlainText(), self.inputopt)

        for c in range(properties.shape[1]):
            for r in range(properties.shape[0]):
                self.infoTable.setItem(r, c, QTableWidgetItem(properties[r, c]))
        print('drawMolecule')

    def similar_mols(self):
        # Collecting text from a inputString
        molecule = self.inputString.toPlainText()

        # if Input String in InChI, change to SMILES
        if self.inputopt == 'InChI':
            mol_ = Chem.inchi.MolFromInchi(molecule)
            molecule = Chem.MolToSmiles(mol_)

        # Create similar molecules, fetch image and data
        self.sim_mols_inchi, self.sim_mols_smiles, self.similar_mols = create_similar_mols_(molecule, self.model_vae,
                                                                                            self.charset, self.stdev)
        print(self.sim_mols_smiles)
        for index, mol in enumerate(self.similar_mols):  # Creating images molecules
            Draw.MolToFile(mol, "temp.svg", size=(800, 800))
            cairosvg.svg2png(url='./temp.svg', write_to='sim_mol' + str(index) + '.png')

        # Setting images for similar moleculew
        similar_mols = [self.similarMol_1, self.similarMol_2, self.similarMol_3, self.similarMol_4,
                        self.similarMol_5, self.similarMol_6, self.similarMol_7, self.similarMol_8,
                        self.similarMol_9]

        for index, mol in enumerate(similar_mols):
            mol.setPixmap(QPixmap('sim_mol' + str(index) + '.png'))

        self.setImagesClick()
        print('similar_mols')

    def setImagesClick(self):
        # Clickable images
        images_click_ = [self.similarMol_1, self.similarMol_2, self.similarMol_3, self.similarMol_4,
                         self.similarMol_5, self.similarMol_6, self.similarMol_7, self.similarMol_8,
                         self.similarMol_9]
        for index, images in enumerate(images_click_):
            images.clicked.connect(partial(self.set_new_smiles, index))
        print('setImagesClick')

    def set_new_smiles(self, index):
        if self.inputopt == 'SMILES':
            self.inputString.setText(self.sim_mols_smiles[index])
        else:
            self.inputString.setText(self.sim_mols_inchi[index])
        self.drawMolecule()
        print('set_new_smiles')

    def prediction_class(self):
        # ML predictions for molecule
        molecule = self.inputString.toPlainText()
        self.res, self.conf = pred_(self.model, molecule, self.inputopt)

        # Results
        self.resultText.setText(self.res + '.')

        # Confidence table
        self.confTable.setEnabled(True)
        for c in range(self.conf.shape[1]):
            for r in range(self.conf.shape[0]):
                self.confTable.setItem(r, c, QTableWidgetItem(self.conf[r, c]))
        print('prediction_class')

    def inputOptions(self):
        # Options for ComboBox
        options = ['SMILES', 'InChI']
        for opt in options:
            self.inputOpt.addItem(opt)

    def chooseInput(self):
        # Changing input type
        self.inputopt = self.inputOpt.currentText()

    def activate_fixLabels(self, state):
        boxes_ = [self.cBox_1, self.cBox_2, self.cBox_3, self.cBox_4, self.cBox_5, self.cBox_6, self.cBox_7,
                  self.cBox_8, self.cBox_9, self.cBox_10, self.cBox_11, self.cBox_12, self.cBox_13, self.cBox_14,
                  self.cBox_15, self.cBox_16, self.cBox_17, self.cBox_18, self.cBox_19, self.cBox_20]
        if state == Qt.Checked:
            for box in boxes_:
                box.setEnabled(True)
            self.plainTextEdit_2.setEnabled(True)

        else:
            for box in boxes_:
                box.setEnabled(False)
            self.plainTextEdit_2.setEnabled(False)

    def createFile(self):
        if os.path.isfile('./fixLabels.xlsx'):
            fixLabels = pd.read_excel(r'fixLabels.xlsx')
        else:
            fixLabels = pd.DataFrame(columns=['Molecule', 'Labels', 'Comments', 'Date'])

        # Creating right labels column for fixLabels
        checkBoxes_ = [self.cBox_1, self.cBox_2, self.cBox_3, self.cBox_4, self.cBox_5, self.cBox_6,
                       self.cBox_7, self.cBox_8, self.cBox_9, self.cBox_10, self.cBox_11, self.cBox_12,
                       self.cBox_13, self.cBox_14, self.cBox_15, self.cBox_16, self.cBox_17, self.cBox_18,
                       self.cBox_19, self.cBox_20]
        fixLabels_string = ''

        for index, cb in enumerate(checkBoxes_):
            if cb.isChecked():
                fixLabels_string += self.conf[index,0] + ', '

        fixLabels_string = fixLabels_string[:-2]

        newData = pd.DataFrame(dict(Molecule=[self.inputString.toPlainText()],
                                    Labels=fixLabels_string,
                                    Comments=[self.plainTextEdit_2.toPlainText()],
                                    Date=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]))
        fixLabels = fixLabels.append(newData, ignore_index=True)
        fixLabels.to_excel(r'fixLabels.xlsx', index=False)


app = QApplication(sys.argv)
widget = MainPage()
widget.show()
sys.exit(app.exec_())
