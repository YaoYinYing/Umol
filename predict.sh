#Predict input

set -e

ID=7NB4
FASTA=./data/test_case/7NB4/7NB4.fasta
TARGET_POS="50,51,53,54,55,56,57,58,59,60,61,62,64,65,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,103,104,124,127,128"
LIGAND_SMILES='CCc1sc2ncnc(N[C@H](Cc3ccccc3)C(=O)O)c2c1-c1cccc(Cl)c1C' #Make sure these are canonical as in RDKit. If you do not have SMILES - you can input a .sdf file to 'make_ligand_feats.py'

OUTDIR=./data/test_case/7NB4/

# configures
NCPU=32
UNICLUST=/mnt/db/uniref30_uc30/uniclust30_2018_08/uniclust30_2018_08
HHBLITS=$(which hhblits)
WEIGHTS_DIR=/mnt/db/weights/Umol/data/params/

POCKET_PARAMS=${WEIGHTS_DIR}/params_pocket.npy #Umol-pocket params
NO_POCKET_PARAMS=${WEIGHTS_DIR}/params_no_pocket.npy #Umol no-pocket params

## Search Uniclust30 with HHblits to generate an MSA (a few minutes)
if [[ ! -f $OUTDIR/$ID'.a3m' ]];then 
    $HHBLITS -i $FASTA -d $UNICLUST -E 0.001 -all -oa3m $OUTDIR/$ID'.a3m' -cpu $NCPU
fi
wait
## Generate input feats (seconds)
umol_make_msa_seq_feats --input_fasta_path $FASTA \
--input_msas $OUTDIR/$ID'.a3m' \
--outdir $OUTDIR

#SMILES. Alt: --input_sdf 'path_to_input_sdf'
umol_make_ligand_feats --input_smiles $LIGAND_SMILES \
--outdir $OUTDIR

wait

POCKET_INDICES=./data/test_case/7NB4/7NB4_pocket_indices.npy #Zero indexed numpy array of what residues are in the pocket (all CBs within 10Å from the ligand)
## Generate a pocket indices file from a list of what residues (zero indexed) are in the pocket (all CBs within 10Å from the ligand). (seconds)
umol_make_targetpost_npy --outfile $POCKET_INDICES --target_pos ${TARGET_POS}

## Predict (a few minutes)
MSA_FEATS=$OUTDIR/msa_features.pkl
LIGAND_FEATS=$OUTDIR/ligand_inp_features.pkl
NUM_RECYCLES=3
#Change to no-pocket params if no pocket
#Then also leave out the target pos
umol_predict --msa_features  $MSA_FEATS \
--ligand_features $LIGAND_FEATS \
--id $ID \
--ckpt_params $POCKET_PARAMS \
--target_pos $POCKET_INDICES \
--num_recycles $NUM_RECYCLES \
--outdir $OUTDIR


wait
RAW_PDB=$OUTDIR/$ID'_pred_raw.pdb'
umol_align_ligand_conformer --pred_pdb $RAW_PDB \
--ligand_smiles $LIGAND_SMILES --outdir $OUTDIR

grep ATOM $OUTDIR/$ID'_pred_raw.pdb' > $OUTDIR/$ID'_pred_protein.pdb'
echo "The unrelaxed predicted protein can be found at $OUTDIR/$ID'_pred_protein.pdb' and the ligand at $OUTDIR/$ID'_pred_ligand.sdf'"


## Relax the protein (a few minutes)
#This fixes clashes mainly in the protein, but also in the protein-ligand interface.

PRED_PROTEIN=$OUTDIR/$ID'_pred_protein.pdb'
PRED_LIGAND=$OUTDIR/$ID'_pred_ligand.sdf'
RESTRAINTS="CA+ligand" # or "protein"
umol_openmm_relax --input_pdb $PRED_PROTEIN \
                        --ligand_sdf $PRED_LIGAND \
                        --file_name $ID \
                        --restraint_type $RESTRAINTS \
                        --outdir $OUTDIR

wait
#Write plDDT to Bfac column
RAW_COMPLEX=$OUTDIR/$ID'_pred_raw.pdb'
RELAXED_COMPLEX=$OUTDIR/$ID'_relaxed_complex.pdb'
umol_add_plddt_to_relaxed  --raw_complex $RAW_COMPLEX \
--relaxed_complex $RELAXED_COMPLEX  \
--outdir $OUTDIR
echo "The final relaxed structure can be found at $OUTDIR/$ID'_relaxed_plddt.pdb'"
