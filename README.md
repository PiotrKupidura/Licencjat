## Basic usage

Pretrained models are available at https://huggingface.co/PiotrKupidura/Licencjat

### Config file

The config.json file should contain following entries
- "dir_read" - directory containing the .npy files created by the create_dataset script, only required for training
- "n" - length of the fragments generated by the model
- "train_batch_size" and "val_batch_size"
- "epochs" - maximal number of training epochs
- "beta_min" and "beta_max" - minimal and maximal value of the loss coefficient corresponding to the KL divergence term, the value for each epochs is determined by cyclic annealing
- "learning_rate"
### Scripts
#### Insert fragments
```
python3 insert_fragments.py -aa -ss -f -s -e -m -r
```
Generates a fragment with the desired amino acid sequence, secondary structure and orientation and inserts it into a PDB file
.
##### Parameters:
- aa - amino acid sequence of the generated fragment (optional, defaults to the sequence from PDB file)
- ss - secondary structure in HEC format (optional, defaults to the secondary structure from PDB file)
- f - path to the PDB file into which the fragment will be inserted
- s - number of the first residue in the fragment
- e - number of the last residue in the fragment
- m - path to the .pt file with the pretrained model
- r - number of fragments to be generated

##### Example usage
```
python3 insert_fragments.py -aa -ss  -f 2VZ5.pdb -s -e -m model.pt -r 50
```
#### Train
```
python3 train.py
```

Trains the model.
#### Create dataset from PDB
```
python3 create_dataset_from_pdb.py -dir_read -dir_write -list_path -len_fragment
```

Extracts the backbone coordinates, amino acid sequence and secondary structure from PDB files and saves them in .npy format.

##### Parameters:
- dir_read - directory containing the PDB files to be parsed
- dir_write - directory into which the .npy files should be saved
- list_path - path to a .txt file containing chain names and their corresponding weights
- len_fragment - length of the created fragments (should be 1 more than the desired fragment length (n in config) as the first 3 atoms are not included in the reconstruction
#### Extract secondary structures
```
python3 extract_structures.py -dir_read -dir_write
```
Extracts all fragments containing only one secondary structures to be used for validation.
##### Parameters
- dir_read - directory containing PDB files to be processed
- dir_write - directory where the resulting .npy files should be saved
#### Validation query
```
python3 validation_query.py -model -n
```
Generates a given number of fragments and for each of them computes the distance (RMSD after superposition) to its closer neighbour from PDB.
##### Parameters:
- model - path to the .pt file with the pretrained model
- dir_read - directory containing the .npy files created by extract_structures.py
- n - number of fragments to be generated
