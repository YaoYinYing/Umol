#!/bin/bash

# run dir
Umol_run_dir=$(readlink -f $(dirname $0))

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [-f FASTA] [-p POCKET_INDICES] [-s LIGAND_SMILES] [-o OUTDIR] [-i ID] [-e PARAMS] [-u UNICLUST] [-n NUM_RECYCLES]"
            exit 0
            ;;
        -f|--fasta)
            FASTA="$2"
            shift 2
            ;;
        -p|--pocket-indices)
            POCKET_INDICES="$2"
            shift 2
            ;;
        -s|--ligand-smiles)
            LIGAND_SMILES="$2"
            shift 2
            ;;
        -o|--outdir)
            OUTDIR="$2"
            shift 2
            ;;
        -i|--id)
            ID="$2"
            shift 2
            ;;
        -e|--params)
            PARAMS="$2"
            shift 2
            ;;
        -u|--uniclust)
            UNICLUST="$2"
            shift 2
            ;;
        -n|--num-recycles)
            NUM_RECYCLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
FASTA=${FASTA:-"./data/test_case/7NB4/7NB4.fasta"}
POCKET_INDICES=${POCKET_INDICES}
LIGAND_SMILES=${LIGAND_SMILES:-'CCc1sc2ncnc(N[C@H](Cc3ccccc3)C(=O)O)c2c1-c1cccc(Cl)c1C'}

OUTDIR=${OUTDIR:-"./Umol_output/"}
OUTDIR=$(readlink -f "$OUTDIR")

ID=${ID:-"default"}

PARAMS=${PARAMS:-"/mnt/db/weights/Umol/params40000.npy"}
UNICLUST=${UNICLUST:-"/mnt/db/uniref30_uc30/uniclust30_2018_08/uniclust30_2018_08"}
NUM_RECYCLES=${NUM_RECYCLES:-3}

mkdir -p ${OUTDIR}

# use traditional way for conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate Umol

# P2Rank
P2Rank=$(which prank)

set -e 

# run msa searching

if [[ ! -f $OUTDIR/$ID'.a3m' ]];then
    HHBLITS=$(which hhblits)
    $HHBLITS -i $FASTA -d $UNICLUST -E 0.001 -all -oa3m $OUTDIR/$ID'.a3m' -cpu 32
fi

python3 $Umol_run_dir/src/make_msa_seq_feats.py --input_fasta_path $FASTA \
--input_msas $OUTDIR/$ID'.a3m' \
--outdir $OUTDIR

python3 $Umol_run_dir/src/make_ligand_feats.py --input_smiles $LIGAND_SMILES \
--outdir $OUTDIR

MSA_FEATS=$OUTDIR/msa_features.pkl
LIGAND_FEATS=$OUTDIR/ligand_inp_features.pkl

RAW_PDB=$OUTDIR/$ID'_pred_raw.pdb'
if [[ ! -f $RAW_PDB ]];then
    python3 $Umol_run_dir/src/predict.py --msa_features  $MSA_FEATS \
    --ligand_features $LIGAND_FEATS \
    --id $ID \
    --ckpt_params $PARAMS \
    --target_pos $POCKET_INDICES \
    --num_recycles $NUM_RECYCLES \
    --outdir $OUTDIR
    wait
fi


python3 $Umol_run_dir/src/relax/align_ligand_conformer.py --pred_pdb $RAW_PDB \
--ligand_smiles $LIGAND_SMILES --outdir $OUTDIR

grep ATOM $OUTDIR/$ID'_pred_raw.pdb' > $OUTDIR/$ID'_pred_protein.pdb'
echo "The unrelaxed predicted protein can be found at ${OUTDIR}/${ID}_pred_protein.pdb' and the ligand at ${OUTDIR}/pred_ligand.sdf"

conda deactivate



conda activate Umol_relax #Assumes you have conda in your path
PRED_PROTEIN=$OUTDIR/$ID'_pred_protein.pdb'
PRED_LIGAND=$OUTDIR/'pred_ligand.sdf'
RESTRAINTS="CA+ligand" # or "protein"

RELAXED_COMPLEX=$OUTDIR/$ID'_relaxed_complex.pdb'
if [[ ! -f $RELAXED_COMPLEX ]];then
    python3 $Umol_run_dir/src/relax/openmm_relax.py --input_pdb $PRED_PROTEIN \
                            --ligand_sdf $PRED_LIGAND \
                            --file_name $ID \
                            --restraint_type $RESTRAINTS \
                            --outdir $OUTDIR
fi
#Deactivate conda - only for the relaxation
conda deactivate

conda activate Umol
RAW_COMPLEX=$OUTDIR/$ID'_pred_raw.pdb'

python3 $Umol_run_dir/src/relax/add_plddt_to_relaxed.py  --raw_complex $RAW_COMPLEX \
--relaxed_complex $RELAXED_COMPLEX  \
--outdir $OUTDIR
echo "The final relaxed structure can be found at ${OUTDIR}/${ID}_relaxed_plddt.pdb"

conda deactivate
