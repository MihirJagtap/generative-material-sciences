import matplotlib.pyplot as plt
from ase import Atoms
from ase.visualize import view
import torch
import numpy as np

file_path = './HYDRA_JOBS/singlerun/2023-08-04/mp_20/eval_opt.pt'
data = torch.load(file_path)

frac_coords = data['frac_coords'].cpu()
atom_types = data['atom_types'].cpu()
lengths = data['lengths'].cpu()
angles = data['angles'].cpu()
num_atoms = data['num_atoms'].cpu()

# Function to construct the unit cell and visualize the material for a given index
def visualize_material(material_index):
    num_atoms_material = num_atoms[0, material_index].item()

    # Get lattice vectors (a, b, c) from lengths and angles
    a, b, c = lengths[0, material_index]
    alpha, beta, gamma = angles[0, material_index]

    # Convert angles from degrees to radians
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    gamma_rad = gamma * (np.pi / 180.0)

    # Calculate Cartesian lattice vectors
    av = np.array([a, 0, 0])
    bv = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    cv = np.array([c * np.cos(beta_rad),
                   c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
                   np.sqrt(c**2 - c * (c * np.cos(beta_rad))**2 / (c * np.cos(beta_rad))**2 - 
                   c * (c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))**2)])

    # Construct the unit cell using lattice vectors
    unit_cell = np.array([av, bv, cv])

    # Create an ASE Atoms object for the specific material
    atoms = Atoms(symbols=atom_types[0, :num_atoms_material],
                  positions=frac_coords[0, :num_atoms_material] @ unit_cell,
                  cell=unit_cell, pbc=True)

    # Visualize the structure using ASE's viewer
    view(atoms)

# Rapidly switch material_index for different materials
# Replace the range with the desired indices you want to visualize
excluded_atom_types = [43,61,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118]  # Add atomic numbers of unwanted elements here


# Rapidly switch material_index for different materials
# Replace the range with the desired indices you want to visualize
for material_index in range(20, 40):
    # Get the atom types for the current material
    atom_types_material = atom_types[0, :num_atoms[0, material_index].item()]

    # Check if any atom type in the material is in the excluded list
    if any(atom_type in excluded_atom_types for atom_type in atom_types_material):
        continue  # Skip visualization for this material

    visualize_material(material_index)