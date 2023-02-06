"""
Author: Jie Li <jerry-li1996@berkeley.edu>
Date created: Feb 4, 2023
"""

from tankbind.feature_utils import extract_torchdrug_feature_from_mol
from tankbind.generation_utils import (get_LAS_distance_constraint_mask,
                                       get_info_pred_distance,
                                       write_with_new_coords)
from tankbind.data import TankBind_prediction
from tankbind.model import get_model
from copy import copy
from rdkit import Chem
import pickle
import pandas as pd
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import os
import shutil
import numpy as np
import argparse
import logging

script_dir = os.path.dirname(os.path.realpath(__file__))
default_model_path = os.path.join(script_dir, "saved_models", "self_dock.pt")

def get_prepared_protein(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data # info, protein_dict

def get_ligands(input_content):
    '''
    input_content can be either a .csv file (assuming a table with two columns: compound_name, sdf_path),
    or a folder where the .sdf files of the ligands are stored.
    '''
    if input_content.endswith(".csv"):
        ligand_info = pd.read_csv(input_content)
        ligand_names = ligand_info["compound_name"].values
        ligand_sdf_list = ligand_info["sdf_path"].values
    else:
        ligand_names = []
        ligand_sdf_list = []
        for file in os.listdir(input_content):
            if file.endswith(".sdf"):
                ligand_names.append(file.split(".")[0])
                ligand_sdf_list.append(os.path.join(input_content, file))
    return ligand_names, ligand_sdf_list

def prepare_ligands(ligand_names, ligand_sdf_list):
    ligand_dict = {}
    ligand_mol_dict = {}
    for name, ligand_sdf in zip(ligand_names, ligand_sdf_list):
        mol = Chem.SDMolSupplier(ligand_sdf)[0]
        ligand_dict[name] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        ligand_mol_dict[name] = mol
    return ligand_dict, ligand_mol_dict

def make_complete_info(info, ligand_names):
    new_info = []
    for i in range(len(ligand_names)):
        compound_info = info.copy()
        compound_info["compound_name"] = ligand_names[i]
        new_info.append(compound_info)
    info = pd.concat(new_info, axis=0)
    info.reset_index(inplace=True, drop=True)
    return info

def make_prediction(model, dataset, bs=10, device="cpu"):
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=False, 
            follow_batch=['x', 'y', 'compound_pair'], num_workers=8)
    affinity_pred_list = []
    y_pred_list = []
    for data in tqdm(data_loader):
        data = data.to(device)
        y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        for i in range(data.y_batch.max() + 1):
            y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())

    affinity_pred_list = torch.cat(affinity_pred_list)
    result_info = dataset.data
    result_info['affinity'] = affinity_pred_list
    selected_result_info = result_info.loc[result_info.groupby(['protein_name', 'compound_name'], \
        sort=False)['affinity'].agg('idxmax')].reset_index()
    return selected_result_info, y_pred_list

def generate_docked_confs(result_info, data, y_pred_list, ligand_mol_dict, output_path, device="cpu"):
    os.makedirs(output_path, exist_ok=True)
    for i, line in result_info.iterrows():
        idx = line['index']
        pocket_name = line['pocket_name']
        ligandName = line['compound_name']
        coords = data[idx].coords.to(device)
        protein_nodes_xyz = data[idx].node_xyz.to(device)
        n_compound = coords.shape[0]
        n_protein = protein_nodes_xyz.shape[0]
        y_pred = y_pred_list[idx].reshape(n_protein, n_compound).to(device)
        y = data[idx].dis_map.reshape(n_protein, n_compound).to(device)
        compound_pair_dis_constraint = torch.cdist(coords, coords)
        mol = copy(ligand_mol_dict[ligandName])
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
        info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, 
                                  LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                  n_repeat=1, show_progress=False)
        new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
        write_with_new_coords(mol, new_coords, os.path.join(output_path, f"{pocket_name}_{ligandName}.sdf"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_data", type=str, required=True)
    parser.add_argument("--ligands", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=default_model_path)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # preparations of protein and ligand data
    info, protein_dict = get_prepared_protein(args.protein_data)
    ligand_names, ligand_sdf_list = get_ligands(args.ligands)
    ligand_dict, ligand_mol_dict = prepare_ligands(ligand_names, ligand_sdf_list)
    info = make_complete_info(info, ligand_names)
    
    # clear any old files and prepare the TankBind_prediction dataset
    shutil.rmtree(os.path.join(args.output_dir, "processed"), ignore_errors=True)
    data = TankBind_prediction(args.output_dir, info, protein_dict, ligand_dict)

    # load model
    logging.basicConfig(level=logging.INFO)
    model = get_model(0, logging, args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    _ = model.eval()

    # make prediction
    prediction_info, y_pred_list = make_prediction(model, data, bs=10, device=args.device)
    prediction_info.to_csv(os.path.join(args.output_dir, "prediction_info.csv"), index=False)
    
    # generate docked conformations
    generate_docked_confs(prediction_info, data, y_pred_list, ligand_mol_dict, args.output_dir, device=args.device)

    print("All finished!")