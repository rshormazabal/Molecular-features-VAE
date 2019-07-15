from molvs.standardize import enumerate_tautomers_smiles
from molvs.tautomer import TautomerEnumerator
import pandas as pd

### enumerate_tautomers_smiles
## Checking tautomers
enumerator = TautomerEnumerator()

tautomers = [enumerator.enumerate(mol) for mol in all_mols]

# IMPORT DATABASE AND LABELS
dataset_raw = pd.read_csv(r'database_inchi.csv')