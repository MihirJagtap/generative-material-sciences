import torch
import numpy as np
from pymatgen import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.local_env import CrystalNN
from math import sqrt
from pymatgen.analysis.structure_matcher import StructureMatcher


# Load data from files
file_path = './HYDRA_JOBS/singlerun/2023-08-04/mp_20/eval_opt.pt'
data = torch.load(file_path)
real_data = torch.load('./HYDRA_JOBS/singlerun/2023-08-04/mp_20/eval_recon.pt')

# Create StructureMatcher instance
custom_comparator = ElementComparator()
sm = StructureMatcher(comparator=custom_comparator)
rmsd_scores = []

# Convert angles to degrees using numpy.degrees()
angles_deg = np.degrees(data['angles'].cpu().numpy())
real_angles_deg = np.degrees(real_data['angles'].cpu().numpy())

# Get the number of available structures
num_structures = 1100
real_num_structures = 5000
print(num_structures)

# Iterate through all available indices
for material_index in range(num_structures):
    # Convert tensor values to numpy arrays
    lattice_lengths = data['lengths'].cpu()
    lattice_angles = data['angles'].cpu()

    # Calculate lattice vectors for generated structure
    a, b, c = lattice_lengths[0, material_index]
    alpha, beta, gamma = lattice_angles[0,material_index]

    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    gamma_rad = gamma * (np.pi / 180.0)

    av = np.array([a, 0, 0])
    bv = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    cv = np.array([c * np.cos(beta_rad),
                   c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
                   np.sqrt(c**2 - c * (c * np.cos(beta_rad))**2 / (c * np.cos(beta_rad))**2 - 
                   c * (c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))**2)])

    lattice_parameters = (a, b, c, alpha, beta, gamma)

    lattice_vectors = Lattice.from_parameters(*lattice_parameters)
    
    # Create structure instances
    atom_types = data['atom_types'].cpu()
    frac_coords = data['frac_coords'].cpu()
    num_atoms = data['num_atoms'].cpu()
    num_atoms_material = num_atoms[0, material_index].item()

    generated_structure = Structure(lattice_vectors, atom_types[0, :num_atoms[0, material_index].item()], frac_coords[0, :num_atoms_material])

    for real_index in range(real_num_structures):
        real_lattice_lengths = real_data['lengths']
        real_lattice_angles = real_data['angles']

        real_a, real_b, real_c = real_lattice_lengths[0, material_index]
        real_alpha, real_beta, real_gamma = real_lattice_angles[0,material_index]
        real_alpha_rad = real_alpha * (np.pi / 180.0)
        real_beta_rad = real_beta * (np.pi / 180.0)
        real_gamma_rad = real_gamma * (np.pi / 180.0)

        real_av = np.array([real_a, 0, 0])
        real_bv = np.array([real_b * np.cos(real_gamma_rad), real_b * np.sin(real_gamma_rad), 0])
        real_cv = np.array([real_c * np.cos(real_beta_rad),
                    real_c * (np.cos(real_alpha_rad) - np.cos(real_beta_rad) * np.cos(real_gamma_rad)) / np.sin(real_gamma_rad),
                    np.sqrt(real_c**2 - real_c * (real_c * np.cos(real_beta_rad))**2 / (real_c * np.cos(real_beta_rad))**2 - 
                    real_c * (real_c * (np.cos(real_alpha_rad) - np.cos(real_beta_rad) * np.cos(real_gamma_rad)) / np.sin(real_gamma_rad))**2)])
        real_lattice_parameters = (real_a, real_b, real_c, real_alpha, real_beta, real_gamma)
        real_lattice_vectors = Lattice.from_parameters(*real_lattice_parameters)

        real_atom_types = data['atom_types']
        real_frac_coords = real_data['frac_coords']
        real_num_atoms = real_data['num_atoms']
        real_num_atoms_material = real_num_atoms[0, material_index].item()

        real_structure = Structure(real_lattice_vectors, real_atom_types[0, :real_num_atoms[0, material_index].item()], real_frac_coords[0, :real_num_atoms_material])

        match = sm.fit(real_structure, generated_structure)

        if match:  # If structures match
            rmsd = match.rms
            rmsd_scores.append(rmsd)
            print(f"RMSD between real {real_index + 1} and generated {material_index + 1} structure: {rmsd:.4f}")
        else:  # If structures don't match
            print(f"Real {real_index + 1} and generated {material_index + 1} structure do not match.")

