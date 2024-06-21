source activate umol

umol \
    input.id=7NB4 \
    input.fasta='./data/test_case/7NB4/7NB4.fasta' \
    "input.ligand.smiles='CCc1sc2ncnc(N[C@H](Cc3ccccc3)C(=O)O)c2c1-c1cccc(Cl)c1C'" \
    input.target_pos=[50,51,53,54,55,56,57,58,59,60,61,62,64,65,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,103,104,124,127,128] \
    output.dir='./data/test_case/7NB4_inference/'
    