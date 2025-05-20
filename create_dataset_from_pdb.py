import numpy as np
from glob import glob
from core.parser import FileParser
import argparse
import os
import tqdm
from multiprocessing import Pool, cpu_count

RESIDUES = {"ALA": 1,  "ARG": 2,  "ASN": 3,  "ASP": 4,
            "CYS": 5,  "GLN": 6,  "GLU": 7,  "GLY": 8,
            "HIS": 9,  "ILE": 10, "LEU": 11, "LYS": 12,
            "MET": 13, "PHE": 14, "PRO": 15, "SER": 16,
            "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20}

STRUCTURES = {"H": 1, "E": 2, "C": 3}

def read_weights(list_path):
    file = open(list_path, "r")
    weights = dict([(line.split()[0][:4], (line.split()[0][4], line.split()[1])) for line in file.readlines()])
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir_read", type=str, required=True)
    parser.add_argument("-dir_write", type=str, required=True)
    parser.add_argument("-list_path", type=str, required=True)
    parser.add_argument("-len_fragment", type=int, required=True) # 1 more than the number of residues in the generated fragments
    args = parser.parse_args()
    dir_read = args.dir_read
    dir_write = args.dir_write
    list_path = args.list_path
    len_fragment = args.len_fragment
    if not os.path.exists(dir_write):
        os.makedirs(dir_write)
    if not os.path.exists(f"{dir_write}/inputs"):
        os.makedirs(f"{dir_write}/inputs")
    if not os.path.exists(f"{dir_write}/labels"):
        os.makedirs(f"{dir_write}/labels")
    files = glob(f"{dir_read}/*.pdb")
    weights = read_weights(list_path)
    pbar = tqdm.tqdm(files)
    def process_file(file):
        file_parser = FileParser(file)
        file_name = file.split('/')[-1].split('.')[0].upper()
        chain, weight = weights[file_name]
        structure = file_parser.load_structure(chain)
        inputs, labels = structure.generate_observations(len_fragment, weight)
        if inputs is not None:
            np.save(f"{dir_write}/inputs/{file.split('/')[-1].split('.')[0]}.npy", inputs)
            np.save(f"{dir_write}/labels/{file.split('/')[-1].split('.')[0]}.npy", labels)
    num_workers = cpu_count()
    with Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.imap(process_file, files), total=len(files)))
