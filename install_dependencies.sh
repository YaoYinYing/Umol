#Install conda env
conda create -n umol -y
conda activate umol
conda install python=3.10 -y
conda install -y ambertools cudatoolkit=11.8.0 cudnn=8.9.2.26 openmm=8.1.0 openmmforcefields=0.11.2 pdbfixer
conda install -y -c bioconda -c conda-forge hhsuite --no-deps


pip install . 
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install numpy==1.26.4


wait


## Get network parameters for Umol (a few minutes)
#Pocket params
wget https://zenodo.org/records/10397462/files/params40000.npy
mkdir data/params
mv params40000.npy  data/params/params_pocket.npy
#No-pocket params
wget https://zenodo.org/records/10489242/files/params60000.npy
mv params60000.npy  data/params/params_no_pocket.npy

wait
## Get Uniclust30 (10-20 minutes depending on bandwidth)
# 25 Gb download, 87 Gb extracted
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
mkdir data/uniclust30
mv uniclust30_2018_08_hhsuite.tar.gz data
cd data
tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
cd ..

wait
