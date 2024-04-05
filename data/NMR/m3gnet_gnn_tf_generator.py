import shutil
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.xyz import XYZ
from pymatgen.core import Lattice, Structure

from m3gnet.models import M3GNet, M3GNetCalculator, Potential

import random
import os
import sys

if __name__ == "__main__":

    # Reading NMR values
    with open("./data/SI_DFT_NMR.txt") as f:
        lines = f.read().splitlines()

    nmr_dict = {}
    for i, row in enumerate(lines):
        if row.isnumeric():
            element_num = int(row)
            dsgdb_id = lines[i + 1]
            nmr_rows = lines[i + 2 : i + 2 + element_num]
            nmr_list = [s.split()[1] for s in nmr_rows]
            nmr_dict[dsgdb_id] = nmr_list

    # Reading atoms geometry file
    xyz = XYZ.from_file("./data/SI_DFT_geo.xyz")
    all_mols = xyz.all_molecules
    total_num = 130831
    dsgdb_list = list(nmr_dict.keys())

    # Setting random seed value
    random_seed_value = 0
    random.seed(random_seed_value)

    # Making directory
    dir_path = "./m3gnet_descriptor"
    os.makedirs(dir_path, exist_ok=True)

    # Getting calculator
    potential = Potential(M3GNet.load())
    calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)

    # Getting all atom descriptor
    df_total = pd.DataFrame()
    for i, mol in enumerate(all_mols):

        # Creating atoms descriptor by m3gnet
        ase_mol = AseAtomsAdaptor.get_atoms(mol)
        ase_mol.set_calculator(calculator)
        ase_mol.get_potential_energy()

        # Getting atoms infomation
        element_num = len(mol)
        molecule_formula = mol.formula
        dsgdb_idx = dsgdb_list[i]
        nmr_list = nmr_dict[dsgdb_idx]

        # Reading atoms descriptor of m3gnet(the name is each_atom_energy.csv)
        df_temp = pd.DataFrame()
        df_temp = pd.read_csv("./each_atom_energy.csv")
        df_temp.insert(1, "nmr_shift", nmr_list)
        df_temp.insert(1, "dsgdb_idx", dsgdb_idx)
        df_temp.insert(1, "molcule_formula", molecule_formula)

        # Adding atoms infomation and descriptor
        df_total = pd.concat([df_total, df_temp], axis=0)

    # Splitting into element type
    df_total_C = df_total[df_total["atomic_number"] == 6.0]
    df_total_H = df_total[df_total["atomic_number"] == 1.0]
    df_total_N = df_total[df_total["atomic_number"] == 7.0]
    df_total_O = df_total[df_total["atomic_number"] == 8.0]
    df_total_F = df_total[df_total["atomic_number"] == 9.0]

    print("---------------------------")
    print(f"Whole record:{len(df_total)}")
    print(f"C total record:{len(df_total_C)}")
    print(f"H total record:{len(df_total_H)}")
    print(f"N total record:{len(df_total_N)}")
    print(f"O total record:{len(df_total_O)}")
    print(f"F total record:{len(df_total_F)}")
    print("---------------------------")

    # Saving m3gnet descriptor into csv file
    df_total_C.to_csv(f"m3gnet_total_C.csv")
    df_total_H.to_csv(f"m3gnet_total_H.csv")
    df_total_N.to_csv(f"m3gnet_total_N.csv")
    df_total_O.to_csv(f"m3gnet_total_O.csv")
    df_total_F.to_csv(f"m3gnet_total_F.csv")
    df_total.to_csv(f"m3gnet_total.csv")
