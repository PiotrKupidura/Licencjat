import numpy as np
from glob import glob
import os
import argparse
from core.parser import FileParser
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir_read", type=str, required=True)
    parser.add_argument("-dir_write", type=str, required=True)
    args = parser.parse_args()
    dir_read = args.dir_read
    dir_write = args.dir_write


    files = glob(f"{dir_read}/*.pdb")
    if not os.path.exists(dir_write):
        os.makedirs(dir_write)
    helices = {}
    sheets = {}
    
    def process_file(file):
        with open(file, "r") as f:
            l = 0
            prev_ss = "-"
            helix = []
            sheet = []
            parser = FileParser(file)
            structure = parser.load_structure()
            for atom in structure._ca:
                ss = atom.ss
                # If the current structure ends, save it
                if ss != prev_ss:
                    if prev_ss == "H":
                        if l in helices:
                            a = np.stack(helix, axis=0)
                            a = a - a[0] # center the structure
                            helices[l].append(a)
                        else:
                            a = np.stack(helix, axis=0)
                            a = a - a[0] # center the structure
                            helices[l] = [a]
                        helix = []
                    elif prev_ss == "S":
                        if l in sheets:
                            a = np.stack(sheet, axis=0)
                            a = a - a[0] # center the structure
                            sheets[l].append(a)
                        else:
                            a = np.stack(sheet, axis=0)
                            a = a - a[0] # center the structure
                            sheets[l] = [a]
                        sheet = []
                    l = 0
                if ss == "H":
                    a = np.array([float(i) for i in atom.coordinates])
                    helix.append(a)
                elif ss == "S":
                    a = np.array([float(i) for i in atom.coordinates])
                    sheet.append(a)
                l += 1
                prev_ss = ss
                
    for file in tqdm.tqdm(files):
        process_file(file)
        
    for key, value in helices.items():
        np.save(f"{dir_write}/helix_{key}.npy", value)
    for key, value in sheets.items():
        np.save(f"{dir_write}/sheet_{key}.npy", value)
    
