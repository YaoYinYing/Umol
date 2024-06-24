# Umol - **U**niversal **mol**ecular framework

## Structure prediction of protein-ligand complexes from sequence information

The protein is represented with a multiple sequence alignment and the ligand as a SMILES string, allowing for unconstrained flexibility in the protein-ligand interface. There are two versions of Umol: one that uses protein pocket information (recommended) and one that does not. Please see the [test case](test.sh) for more information.

[Read the paper here](https://www.nature.com/articles/s41467-024-48837-6)

<img src="./Network.svg"/>

Umol is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). \
The Umol parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode).

# Colab (run Umol in the browser)

[Colab Notebook](https://colab.research.google.com/github/YaoYinYing/Umol/blob/main/Umol.ipynb)

# Local installation

## (several minutes)

The entire installation takes <1 hour on a standard computer. \
We assume you have CUDA12. For CUDA11, you will have to change the installation of some packages. \
The runtime will depend on the GPU you have available and the size of the protein-ligand complex you are predicting. \
On an NVIDIA A100 GPU, the prediction time is a few minutes on average.

Ensure that you have miniconda installed, see: <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html> or <https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html>

```shell
conda create -n umol -y
conda activate umol
conda install python=3.10 -y
conda install -y ambertools cudatoolkit=11.8.0 cudnn=8.9.2.26 openmm=8.1.0 openmmforcefields=0.11.2 pdbfixer
conda install -y -c bioconda -c conda-forge hhsuite --no-deps

pip install . 
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install numpy==1.26.4
```

## Get Uniclust30 (10-20 minutes depending on bandwidth) if you need to build MSA on local machine

# 25 Gb download, 87 Gb extracted

```shell
UC30_DB_DIR=/path/to/uniclust30
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
mkdir -p $UC30_DB_DIR
mv uniclust30_2018_08_hhsuite.tar.gz data
pushd $UC30_DB_DIR
    tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
popd
```

## Fetch the pretrained weights

```shell
UMOL_WEIGHTS_DIR=/path/to/weights
#pocket params
wget -q https://zenodo.org/records/10397462/files/params40000.npy -O $UMOL_WEIGHTS_DIR/params_pocket.npy

#No-pocket params
wget -q https://zenodo.org/records/10489242/files/params60000.npy -O $UMOL_WEIGHTS_DIR/params_no_pocket.npy
```

# Run the test case

## (a few minutes)

```shell
source activate umol

umol \
    input.id=7NB4 \
    input.fasta='./data/test_case/7NB4/7NB4.fasta' \
    "input.ligand.smiles='CCc1sc2ncnc(N[C@H](Cc3ccccc3)C(=O)O)c2c1-c1cccc(Cl)c1C'" \
    input.target_pos=[50,51,53,54,55,56,57,58,59,60,61,62,64,65,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,103,104,124,127,128] \
    weights.dir=$UMOL_WEIGHTS_DIR \
    database.uc30=$UC30_DB_DIR \
    output.dir='./data/test_case/7NB4_inference/'
```

## Run Umol from Python code

**An example from Colab Notebook**

```python
# necessary imports
import os
import shutil
import hydra
import os
import umol.inference
from umol.inference import main
from omegaconf import DictConfig, OmegaConf

# remove output directory if it exists
if os.path.exists(OUTDIR):
  shutil.rmtree(OUTDIR)

# copy provided msa to output directory
os.makedirs(os.path.join(OUTDIR, 'msas'), exist_ok=True)
shutil.copy(os.path.join(f'{ID}.a3m'), os.path.join(OUTDIR, 'msas', 'output.a3m'))

mocked_db='/content/mock/uniref30_uc30/uniclust30_2018_08/uniclust30_2018_08_mock.file'

# if you need to skip msa building by providing a pre-computed a3m file, please mockout the uc30 database path if it does not exist
if not os.path.exists((mocked_db_dir:=os.path.dirname(mocked_db))):
  # here we mock out ur30 database bcs we have already uploaded the msa file.
  os.makedirs(mocked_db_dir, exist_ok=True)

  with open(mocked_db, 'w'): 
    ...

# Path of the configure file
cfg_path=os.path.join(os.path.abspath(os.path.dirname(umol.inference.__file__)), 'config')

# Instantialize Hydra 
try:
  hydra.initialize_config_dir(
      version_base=None, config_dir=cfg_path
  )
except ValueError as e :
  print(f'Ignore re-instantializing Hydra: {e}')

def reload_config_file(config_name: str = 'umol') -> DictConfig:
    return hydra.compose(
        config_name=config_name,
        return_hydra_config=False,
    )

# parse the global configuration yaml
cfg=reload_config_file()

updated_cfg = {'input.fasta': FASTA_FILE,
  'input.ligand.smiles': LIGAND,
  'input.target_pos': TARGET_POSITIONS,
  'weights.dir': UMOL_WEIGHTS_DIR,
  'runtime.recycles': NUM_RECYCLES,
  'input.id': ID,
  'output.dir': OUTDIR,
  # also mocked uc30 db path
  'database.uc30': mocked_db[:-10]
}


# update the configs with all the inputs
for k, v in updated_cfg.items():
  OmegaConf.update(cfg, k, v)


# run with updated configuration
main(cfg)
```

## Extract target positions from a pdb file of your choice

```shell
PDB_FILE=./data/test_case/7NB4/7NB4.pdb1
PROTEIN_CHAIN='A'
LIGAND_NAME='U6Q'
OUTDIR=./data/test_case/7NB4/
python3 ./src/parse_pocket.py --pdb_file $PDB_FILE \
--protein_chain $PROTEIN_CHAIN \
--ligand_name $LIGAND_NAME \
--outdir $OUTDIR
```

# Citation

Bryant, P., Kelkar, A., Guljas, A. Clementi, C. and Noé F.
Structure prediction of protein-ligand complexes from sequence information with Umol. Nat Commun 15, 4536 (2024). <https://doi.org/10.1038/s41467-024-48837-6>
