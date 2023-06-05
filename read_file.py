# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:16:55 2023

@author: marlijn 

Used for steps 
https://xinhaoli74.github.io/blog/rdkit/2021/01/06/rdkit.html#Setup

"""

# import packages 
import pandas as pd
import rdkit

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools

rdkit.__version__

# Read file and set in list 
df = pd.read_excel('')                  # df = data frame

# add molecules to list 
PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles')
df.head(1)

# Number of atoms 
df["n_Atoms"] = df['ROMol'].map(lambda x: x.GetNumAtoms())
df.head(1)

######## EIGENSCHAOOEN DIE WE WILLEN GEBRUIKEN OM TE CHECKEN TOEVOEGEN


# Find groups in molecule
m = Chem.MolFromSmiles('c1ccccc1O')              # Find structure
patt = Chem.MolFromSmarts('ccO')                 # 
m.HasSubstructMatch(patt)                        

patt = Chem.MolFromSmarts('c1ccncn1')
patt

matches = [m for m in df['ROMol'] if m.HasSubstructMatch(patt)]
print(f'There are {len(matches)} matched molecules')
matches[0]

######################## PRE PROCESSING DATA ##################################

# Find missing data and remove row/collum
missing_values = df.isna().sum().sum()     
if missing_values > 0: 
    print('Remove missing values')
else: 
    print('No missing_values')
    
    
# Find duplicated rows if any
duplicate = df.duplicated(keep=False)
duplicates = df.duplicated(keep=False).sum()

if duplicates > 0:
    print('Remove duplicates')
else: 
    print('No duplicates')
    
# Find correlation, show with figure
correlation=round(df.corr(),2)            #kijken of eerste 3 kolommen er nog af kunnen. 
correlation.style.background_gradient(cmap='BrBG_r', axis=None)



# # Drop molecules before saving list as file 
# data = data.drop(['ROMol'], axis=1)
# data.head(1)

# # Get molecule from smiles
# smiles = 'COC(=O)c1c[nH]c2cc(OC(C)C)c(OC(C)C)cc2c1=O'  # Nog schrijven vanuit excel
# mol = Chem.MolFromSmiles(smiles)