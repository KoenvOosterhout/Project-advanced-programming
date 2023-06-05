# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:16:55 2023

@author: marlijn 

<<<<<<< Updated upstream
Used for steps 
https://xinhaoli74.github.io/blog/rdkit/2021/01/06/rdkit.html#Setup
=======
Can be used for parameters:
https://xinhaoli74.github.io/blog/rdkit/2021/01/06/rdkit.html#Setup
https://www.rdkit.org/
>>>>>>> Stashed changes

"""

# import packages 
import pandas as pd
<<<<<<< Updated upstream
import rdkit
=======
import matplotlib.pyplot as plt
import rdkit
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
>>>>>>> Stashed changes

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools

rdkit.__version__

<<<<<<< Updated upstream
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
=======

def adding_descriptors(df): 
    '''  Adding  descriptors to the dataframe 
   
    Parameters
    ----------
    df : dataframe where descriptors are added

    Returns
    -------
    None.
    '''

    # add molecules to list 
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')
    df.head(1)

    # Number of atoms 
    df["n_Atoms"] = df['ROMol'].map(lambda x: x.GetNumAtoms())
    df.head(1)
    
    ######## EIGENSCHAOOEN DIE WE WILLEN GEBRUIKEN OM TE CHECKEN TOEVOEGEN

    # Find groups in molecule
        
    # Using SMARTS instead of SMILES, 
    #SMILES describe molecules and SMARTS describe patterns
    patt = Chem.MolFromSmarts('c1ccncn1')
    patt

    matches = [m for m in df['ROMol'] if m.HasSubstructMatch(patt)]
    print(f'There are {len(matches)} matched molecules')
    matches[0]
    
    return df


def pre_processing(df):
    '''
    Pre processing the data by checking missing values, finding
    
    Parameters
    ----------
    df : dataframe that is pre processed
    
    Returns
    -------
    None
    '''    
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


    # # Find correlation                       WERKT NOG NIET
    # correlation=round(df.corr(),2)             
    # correlation.style.background_gradient(cmap='BrBG_r', axis=None)
    
    return df
 
    
def scaling(df):                          # WILLEN WE MINMAX OF STANDARD?
    
    # Drop molecules before saving list as file 
    df = df.drop(['ROMol'], axis=1)
    df_1.head(1)
    
    df = df.drop(['SMILES'], axis=1)
    df.head(1)
            
    # MinMax                  
    # df_minmax = MinMaxScaler().fit_transform(df)
    df_minmax = (df - df.min()) / (df.max() - df.min())

    return df_minmax

    # # Standard   
    # df_standard = StandardScaler().fit_transform(df)
    # df_standard  = (df - df.mean()) / df.std()

# def plot_scaling(df_scaled):               PLOTTEN WERKT NOG NIET 
#     df_scaled.plot(kind='box', figsize=(9,6))
#     plt.title('Min Max scaling 2', fontsize=20);


# Read file
df_1 = pd.read_excel('tested_molecules_1.xlsx')                 # df = data frame
df_2 = pd.read_excel('tested_molecules_2.xlsx') 

# Adding descriptors 
df_1 = adding_descriptors(df=df_1)
df_2 = adding_descriptors(df=df_2)

# Pre processing
df_1 = pre_processing(df=df_1)
df_2 = pre_processing(df=df_2)  

# Scaling
df_1_scaled = scaling(df=df_1)
df_2_scaled = scaling(df=df_2)

# # Plot                         PLOTTEN WERKT NOG NIET
# df_1_plot = plot_scaling(df_scaled = df_1_scaled)
# df_2_plot = plot_scaling(df_scaled = df_2_scaled)




>>>>>>> Stashed changes
