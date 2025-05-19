import numpy as np
from glob import glob
import argparse

RESIDUES = {"ALA": 1,  "ARG": 2,  "ASN": 3,  "ASP": 4,
            "CYS": 5,  "GLN": 6,  "GLU": 7,  "GLY": 8,
            "HIS": 9,  "ILE": 10, "LEU": 11, "LYS": 12,
            "MET": 13, "PHE": 14, "PRO": 15, "SER": 16,
            "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20}

STRUCTURES = {"H": 1, "E": 2, "C": 3}

def read_weights():
    file = open("data/7/lista.txt", "r")
    weights = dict([(line.split()[0], line.split()[1]) for line in file.readlines()])
    return weights


def parse_line(line:str):
    x = line.split()
    inputs = np.array([x[5:8]], dtype=np.float32)
    ss = STRUCTURES[x[4]]
    res = RESIDUES[x[1]] if x[1] in RESIDUES else 0
    aa = np.array(res, dtype=np.float32)
    labels = np.stack([aa, ss], axis=0)
    return inputs, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_read", type=str)
    parser.add_argument("--dir_write", type=str)
    parser.add_argument("--len_fragment", type=int) # 1 more than the number of residues in the generated fragments
    args = parser.parse_args()
    dir_read = args.dir_read
    dir_write = args.dir_write
    len_fragment = args.len_fragment
    files = glob(f"{dir_read}/*.dat")
    weights = read_weights()
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        name = name.replace("_", "").upper()
        weight = weights[name]
        inputs, labels = [], []
        lines = [line for line in open(file, "r").readlines()]
        i = 0
        # Move the window by 1 residue at a time
        while i < (len(lines) - len_fragment + 1):
            inp = []
            lab = []
            gap = False
            # For each residue in the window
            for j in range(len_fragment):
                # If the residue is a gap, move the window by the number of residues in the gap
                if lines[i+j].split()[-1] == "GAP":
                    gap = True
                    i += j
                    break
                inp.append([parse_line(lines[i+j])[0]])
                lab.append([np.concatenate([parse_line(lines[i+j])[1], np.array([weight])], axis=0)])
            # If the window does not contain a gap, add the input and label to the list
            if not gap:
                inp = np.stack(inp, axis=0).squeeze()
                lab = np.stack(lab, axis=0).squeeze()
                displacement = inp[-1,:] - inp[2,:]
                lab = np.concatenate([lab, np.broadcast_to(np.expand_dims(displacement, axis=0), (lab.shape[0],3))], axis=-1)
                inputs.append(inp)
                labels.append(lab)
            # Move the window by 1 residue
            i += 1
        # Convert the list of inputs and labels to numpy arrays
        inputs = np.array(inputs)
        labels = np.array(labels, dtype=np.float32)
        inputs_file = f"{dir_write}/inputs/{file.split('/')[-1].split('.')[0]}.npy"
        labels_file = f"{dir_write}/labels/{file.split('/')[-1].split('.')[0]}.npy"
        inputs = inputs.squeeze()
        labels = labels.squeeze()
        if len(inputs.shape) != 3 or len(labels.shape) != 3:
            print(name)
            continue
        np.save(inputs_file, inputs)
        np.save(labels_file, labels)
