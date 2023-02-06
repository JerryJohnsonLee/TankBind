"""
Author: Jie Li <jerry-li1996@berkeley.edu>
Date created: Feb 4, 2023
"""

from tankbind.feature_utils import get_protein_feature
from Bio.PDB import PDBParser
import pickle
import os
import pandas as pd
import numpy as np
import argparse
import shutil

parser = PDBParser(QUIET=True)

p2rank_cmd = "bash /home/jerry/src/p2rank_2.4/prank"
tmp_dir = "/tmp"

def prepare_protein(protein_pdb, center=None, max_radius=20):
    global p2rank_cmd
    global tmp_dir

    pdb_name = os.path.basename(protein_pdb).split(".")[0]
    shutil.copy(protein_pdb, os.path.join(tmp_dir, f"{pdb_name}.pdb"))

    # run p2rank to get all pockets information
    with open(os.path.join(tmp_dir, "proteins.ds"), "w") as f:
        f.write(f"{pdb_name}.pdb")
    os.system(f"{p2rank_cmd} predict {tmp_dir}/proteins.ds -o {tmp_dir}/p2rank_out -threads 1")

    # get p2rank output
    p2rank_pockets = pd.read_csv(os.path.join(tmp_dir, "p2rank_out", f"{pdb_name}.pdb_predictions.csv"))
    p2rank_pockets.columns = p2rank_pockets.columns.str.strip()
    pocket_coms = p2rank_pockets[['center_x', 'center_y', 'center_z']].values
    if center is not None:
        distances = np.linalg.norm(pocket_coms - center, axis=1)
        closest_idx = distances.argmin()
        if distances[closest_idx] > max_radius:
            raise ValueError("No pocket found within max_radius of the given center!")
        pocket_coms = pocket_coms[closest_idx][None]

    # protein features
    struc = parser.get_structure('protein', protein_pdb)
    residues = list(struc.get_residues())
    protein_dict = {}
    features = get_protein_feature(residues)
    protein_dict["protein"] = features

    # prepare protein information used as TankBind input
    info = []
    com = ",".join([str(a.round(3)) for a in protein_dict["protein"][0].mean(axis=0).numpy()])
    info.append(["protein", "compound", "protein_center", com])

    for ith_pocket, com in enumerate(pocket_coms):
        com = ",".join([str(a.round(3)) for a in com])
        info.append(["protein", "compound", f"pocket_{ith_pocket+1}", com])
    info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
   
    return info, protein_dict


def parse_args():
    global p2rank_cmd
    global tmp_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_pdb", type=str, required=True)
    parser.add_argument("--center", type=str, default=None)
    parser.add_argument("--max_radius", type=float, default=20)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--p2rank_cmd", type=str, default=p2rank_cmd)
    parser.add_argument("--tmp_dir", type=str, default=tmp_dir)
    args = parser.parse_args()

    p2rank_cmd = args.p2rank_cmd
    tmp_dir = args.tmp_dir
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.center is not None:
        args.center = np.array([float(a) for a in args.center.split(",")])
    info, protein_dict = prepare_protein(args.protein_pdb, args.center, args.max_radius)
    
    with open(os.path.join(args.output_dir, "protein.pkl"), "wb") as f:
        pickle.dump([info, protein_dict], f)

    print("Protein processing complete!")