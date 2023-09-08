from ase.visualize import view
from pymatgen import Structure
import pandas as pd
import json
from pymatgen.io.ase import AseAtomsAdaptor

file_path = 'csv_structure_data.csv'
df_generated_data = pd.read_csv(file_path)

specific_index = 76  # Replace with the index you want to visualize

generated_row = df_generated_data.loc[specific_index]
generated_cif_data = generated_row['cif']
generated_cif_dict = json.loads(generated_cif_data)
generated_structure = Structure.from_dict(generated_cif_dict)

# Visualize the structure using pymatgen's StructureVis
ase_atoms = AseAtomsAdaptor().get_atoms(generated_structure)

view(ase_atoms)

