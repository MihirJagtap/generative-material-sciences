from pymatgen import Lattice, Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.local_env import CrystalNN
from math import sqrt
import pandas as pd 
import json



file_path = 'csv_structure_data.csv'
df_generated_data = pd.read_csv(file_path)
csv_real_data_path = "./data/mp_20/test.csv"
df_real_data = pd.read_csv(csv_real_data_path)


# Create StructureMatcher instance
custom_comparator = ElementComparator()
sm = StructureMatcher(primitive_cell=False)

# Convert angles to degrees using numpy.degrees()
# Get the number of available structures
num_structures = df_generated_data.shape[0]
real_num_structures = df_real_data.shape[0]   

for material_index, generated_row in df_generated_data.iterrows():
    count = 0
    generated_cif_data = generated_row['cif']
    generaeted_cif_dict = json.loads(generated_cif_data)
    generated_structure = Structure.from_dict(generaeted_cif_dict)
    print(generated_structure)

    for real_index, real_row in df_real_data.iterrows():
    # Access the CIF data from the column
        real_cif_data = real_row['cif']

    # Parse the CIF data using CifParser from pymatgen
        cif_parser = CifParser.from_string(real_cif_data)
        real_structure = Structure.from_sites(cif_parser.get_structures()[0].sites)
    
    # Process the real data and compare with generated structure (no change here)
    
        match = sm.fit(real_structure, generated_structure)
        if match:
            count = count + 1

    print(f"material {material_index + 1} had {count} matches")