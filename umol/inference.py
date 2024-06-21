import os
import pickle
import shutil
import warnings

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd

from umol.net.data.tools.utils import timing_decorator
from umol.make_msa_seq_feats import process_msa_to_feature
from umol.make_ligand_feats import bonds_from_smiles, sdf_to_smiles
from umol.net.data.tools import hhblits
from umol.net.model import config
from umol.relax.add_plddt_to_relaxed import read_pdb_plddt, write_pdb_plddt
from umol.relax.align_ligand_conformer import align_coords_transform, generate_best_conformer, read_pdb, write_sdf
from umol.relax.openmm_relax import relax

script_path = os.path.abspath(os.path.dirname(__file__))


def check_pretrained_weights(weights_dir: str, weight_file: str) -> str:

    weight_path=os.path.join(weights_dir, weight_file)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f'Weight file "{weight_path}" does not exist')
    

    return weight_path

@timing_decorator("HHblits")
def run_msa(
    bin_path: str, database_path: str, nproc: int, input_fasta_path: str, outdir: str
) -> str:
    runner = hhblits.HHBlits(
        binary_path=bin_path,
        databases=[database_path],
        query_dir=outdir,
        n_cpu=nproc,
        E_value=0.001,
        all_seqs=True,
    )
    return runner.query(input_fasta_path)


@timing_decorator("Making MSA features")
def make_msa_seq_feats(input_fasta_path: str, input_msas: list, outdir: str) -> str:

    # Get feats
    feature_dict = process_msa_to_feature(input_fasta_path, input_msas)

    # Write out features as a pickled dictionary.

    feat_dir = os.path.join(outdir, "features")
    os.makedirs(feat_dir, exist_ok=True)
    features_output_path = os.path.join(feat_dir, "msa_features.pkl")
    with open(features_output_path, "wb") as f:
        pickle.dump(feature_dict, f, protocol=4)
    print("Saved MSA features to", features_output_path)
    return features_output_path



@timing_decorator("Making Ligand features")
def make_ligand_feats(
    outdir: str, input_smiles: str = None
) -> str:
    if not input_smiles:
        raise ValueError("No input ligand provided")

    feat_dir = os.path.join(outdir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    # Atom encoding - no hydrogens
    atom_encoding = {
        "B": 0,
        "C": 1,
        "F": 2,
        "I": 3,
        "N": 4,
        "O": 5,
        "P": 6,
        "S": 7,
        "Br": 8,
        "Cl": 9,  # Individual encoding
        "As": 10,
        "Co": 10,
        "Fe": 10,
        "Mg": 10,
        "Pt": 10,
        "Rh": 10,
        "Ru": 10,
        "Se": 10,
        "Si": 10,
        "Te": 10,
        "V": 10,
        "Zn": 10,  # Joint (rare)
    }

    # Get the atom types and bonds
    atom_types, atoms, bond_types, bond_lengths, bond_mask = bonds_from_smiles(
        input_smiles, atom_encoding
    )

    ligand_inp_feats = {}
    ligand_inp_feats["atoms"] = atoms
    ligand_inp_feats["atom_types"] = atom_types
    ligand_inp_feats["bond_types"] = bond_types
    ligand_inp_feats["bond_lengths"] = bond_lengths
    ligand_inp_feats["bond_mask"] = bond_mask
    # Write out features as a pickled dictionary.

    features_output_path = os.path.join(feat_dir, "ligand_inp_features.pkl")
    with open(features_output_path, "wb") as f:
        pickle.dump(ligand_inp_feats, f, protocol=4)
    print("Saved features to", features_output_path)

    return features_output_path


def make_pocket_indice(target_pos: list[int],outdir: str) -> str:
    target_position = np.array(target_pos)

    pocket_dir = os.path.join(outdir, "features")
    os.makedirs(pocket_dir, exist_ok=True)
    pocket_file=os.path.join(pocket_dir, 'pocket_indices.npy')
    np.save(pocket_file, target_position)
    return pocket_file


@timing_decorator("Prediction")
def run_prediction(config: config.CONFIG, msa_features,
                ligand_features: str,
                id: str,
                target_pos: str,
                ckpt_params: str,
                num_recycles: int,
                outdir: str) -> str:
    
    from umol.predict import predict
    
    predicted_pdb=predict(config,
                msa_features,
                ligand_features,
                id,
                target_pos,
                ckpt_params,
                num_recycles,
                outdir=outdir)
    
    return predicted_pdb
    
@timing_decorator("Alignment of ligand comformer")
def align_ligand_comformer(job_id:str,raw_pred_pdb: str, ligand_smiles: str,outdir: str):
    pred_ligand = read_pdb(raw_pred_pdb)
    outdir = outdir

    #Get a nice conformer
    best_conf, best_conf_pos, best_conf_err, atoms, nonH_inds, mol, best_conf_id  = generate_best_conformer(pred_ligand['chain_coords'], ligand_smiles)
    #Save error
    conf_err = pd.DataFrame()
    conf_err['id'] = [job_id]
    conf_err['conformer_dmat_err'] = best_conf_err
    conf_err.to_csv(os.path.join(outdir,'conformer_dmat_err.csv'), index=None)
    #Align it to the prediction
    aligned_conf_pos = align_coords_transform(pred_ligand['chain_coords'], best_conf_pos, nonH_inds)

    sdf_save_dir=os.path.join(outdir,'sdf')
    os.makedirs(sdf_save_dir, exist_ok=True)
    sdf_path=os.path.join(sdf_save_dir,conf_err['id'].values[0]+'_pred_ligand.sdf')

    #Write sdf - better to define bonds
    write_sdf(mol, best_conf, aligned_conf_pos, best_conf_id, sdf_path )
    return sdf_path


def unrelaxed_protein(predicted_raw_pdb:str)-> str:
    dir=os.path.dirname(predicted_raw_pdb)
    stem=os.path.basename(predicted_raw_pdb)[:-8]
    unrelaxed_protein_path=os.path.join(dir,f'{stem}_protein.pdb')

    # Open the input and output files
    with open(predicted_raw_pdb, 'r') as infile, open(unrelaxed_protein_path, 'w') as outfile:
        # Copy lines starting with 'ATOM' to the output file
        for line in infile:
            if line.startswith('ATOM'):
                outfile.write(line)

    return unrelaxed_protein_path


def add_plddt_to_comlex(job_id:str,input_raw_complex: str,input_relaxed_complex:str, outdir: str) -> str:
    #Data
    raw_coords, raw_chains, raw_atom_numbers, raw_3seq, raw_resnos, raw_atoms, raw_bfactors = read_pdb_plddt(input_raw_complex)
    relaxed_coords, relaxed_chains, relaxed_atom_numbers,  relaxed_3seq, relaxed_resnos, relaxed_atoms, relaxed_bfactors = read_pdb_plddt(input_relaxed_complex)


    #Write PDB
    relax_dir=os.path.join(outdir, 'relax')
    os.makedirs(relax_dir,exist_ok=True)

    outname=os.path.join(relax_dir,f'{job_id}_relaxed_plddt.pdb')
    write_pdb_plddt(relaxed_coords, relaxed_chains, relaxed_atom_numbers, relaxed_3seq, relaxed_resnos, relaxed_atoms, raw_bfactors, outname)

    return outname


@hydra.main(config_path=os.path.join(script_path, "config"), config_name="umol", version_base=None)
def main(cfg: DictConfig) -> None:

    
    #JAX will preallocate 90% of currently-available GPU memory when the first JAX operation is run.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = f'{str(bool(cfg.runtime.PREALLOCATE)).lower()}'

    os.makedirs(cfg.output.dir, exist_ok=True)
    hhblits_bin_cfg = cfg.bin.hhblits
    fasta_abspath=os.path.abspath(cfg.input.fasta)
    msa_path = run_msa(
        bin_path=hhblits_bin_cfg if hhblits_bin_cfg else shutil.which("hhblits"),
        database_path=cfg.database.uc30,
        nproc=cfg.runtime.nproc,
        input_fasta_path=fasta_abspath,
        outdir=cfg.output.dir,
    )

    msa_feat_path = make_msa_seq_feats(
        input_fasta_path=fasta_abspath, input_msas=(msa_path,), outdir=cfg.output.dir
    )

    if not (ligand_smiles:=cfg.input.ligand.smiles) and (ligand_sdf:=cfg.input.ligand.sdf):
        if not os.path.exists(ligand_sdf):
            raise FileNotFoundError(f'Ligand SDF file "{ligand_sdf}" does not exist')
        ligand_smiles = sdf_to_smiles(ligand_sdf)

    ligand_feat_path = make_ligand_feats(
        outdir=cfg.output.dir,
        input_smiles=ligand_smiles,
    )

    if cfg.input.target_pos:
        pocket_indices = make_pocket_indice(cfg.input.target_pos,cfg.output.dir)
        ckpt_params_file='params_pocket.npy'
    else: 
        pocket_indices = None
        ckpt_params_file='params_no_pocket.npy'

    ckpt_params_path=check_pretrained_weights(cfg.weights.dir,ckpt_params_file)

    raw_predicted_pdb=run_prediction(
        config=config.CONFIG,
        msa_features=msa_feat_path,
        ligand_features=ligand_feat_path,
        id=cfg.input.id,
        target_pos=np.load(pocket_indices) if pocket_indices is not None else [],
        ckpt_params=np.load(ckpt_params_path, allow_pickle=True),
        num_recycles=cfg.runtime.recycles,
        outdir=cfg.output.dir)


    unrelaxed_ligand_path=align_ligand_comformer(job_id=cfg.input.id,
                           raw_pred_pdb=raw_predicted_pdb,
                           ligand_smiles=ligand_smiles,
                           outdir=cfg.output.dir)
    
    unrelaxed_protein_path=unrelaxed_protein(raw_predicted_pdb)

    

    relaxed_complex_path=relax(
        input_pdb=unrelaxed_protein_path,
        outdir=cfg.output.dir,
        mol_in=unrelaxed_ligand_path,
        file_name=cfg.input.id,
        relax_protein_first=cfg.relax.protein_first,
        restraint_type=cfg.relax.restraint_type
          )
    
    final_complex_path_plddt=add_plddt_to_comlex(job_id=cfg.input.id,
                                                input_raw_complex=raw_predicted_pdb,
                                                input_relaxed_complex=relaxed_complex_path,
                                                outdir=cfg.output.dir)
    
    print(f'Final complex with PLDDT scores: {final_complex_path_plddt}')
    
