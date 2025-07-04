import os
import argparse
import logging
import numpy as np
import torch
from tabulate import tabulate
from core.model import CVAE
from core.parser import FileParser, Structure, Atom, MissingO
import matplotlib.pyplot as plt
import json

# logging.getLogger("tensorflow").disabled=True
# logging.getLogger("h5py._conv").disabled=True

RESIDUES = {"A": 1,  "R": 2,  "N": 3,  "D": 4,
            "C": 5,  "Q": 6,  "E": 7,  "G": 8,
            "H": 9,  "I": 10, "L": 11, "K": 12,
            "M": 13, "F": 14, "P": 15, "S": 16,
            "T": 17, "W": 18, "Y": 19, "V": 20}

STRUCTURES = {"H": 1, "E": 2, "C": 3}

def parse_config(config):
    config_file = open(config, "r")
    parameters = json.loads(config_file.read())
    length = parameters["n"]
    latent_dim = parameters["latent_dim"]
    return length, latent_dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-aa", type=str, help="amino acids sequence")
    parser.add_argument("-ss", type=str, help="secondary structure")
    parser.add_argument("-f", "--file", type=str, help="PDB file", required=True)
    parser.add_argument("-s", "--start", type=int, help="initial residue", required=True)
    parser.add_argument("-e", "--end", type=int, help="terminal residue", required=True)
    parser.add_argument("-m", "--model", type=str, help="model to be used", required=True)
    parser.add_argument("-r", "--repeats", type=int, help="number of returned fragments", required=True)
    args = parser.parse_args()

    pdb = args.file
    start = args.start
    end = args.end
    model_path = args.model
    repeats = args.repeats

    input_structure = FileParser(file=pdb).load_structure()

    if args.aa == None:
        aa = input_structure.read_sequence(start-1, end)
    else:
        aa = args.aa

    if args.ss == None:
        ss = input_structure.read_secondary_structure(start-1, end)
    else:
        ss = args.ss
    # aa_1 = torch.tensor([0 for res in aa]).unsqueeze(0).expand(repeats,-1).long()
    aa_1 = torch.tensor([RESIDUES[res] for res in aa]).unsqueeze(0).expand(repeats,-1).long()
    ss_1 = torch.tensor([STRUCTURES[s] for s in ss]).unsqueeze(0).expand(repeats,-1).long()
    if args.aa is None:
        aa_1 = aa_1[:,1:]
        ss_1 = ss_1[:,1:]
        aa = aa[1:]
        ss = ss[1:]
    if len(aa) != end-start+1:
        raise ValueError(f"The number of residues in the input sequence ({len(aa)}) does not match the desired fragment length ({end-start+1})")
    if len(ss) != end-start+1:
        raise ValueError(f"The number of residues in the input secondary structure ({len(ss)}) does not match the desired fragment length ({end-start+1})")

    displacement = input_structure.local_displacement(end,start-1)
    displacement = torch.tensor(displacement).float().unsqueeze(0).expand(repeats,-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, latent_dim = parse_config("config.json")
    if n != end-start+1:
        raise ValueError(f"The fragment length of the model ({n}) does not match the desired fragment length ({end-start+1})")
    model = CVAE(n, latent_dim, 0, 0, 0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # bound atoms not included in rebuilt fragment
    c_1 = input_structure._n[input_structure.find_residue(start-1)].coordinates
    c_2 = input_structure._ca[input_structure.find_residue(start-1)].coordinates
    c_3 = input_structure._c[input_structure.find_residue(start-1)].coordinates

    prev_three = torch.stack([torch.tensor(c_1),torch.tensor(c_2),torch.tensor(c_3)]).unsqueeze(0).expand(repeats,-1,-1).float()
    fragments = model.generate(repeats, prev_three.to(device), aa_1.to(device), ss_1.to(device), displacement.to(device))

    new_structures = [] # all structures obtained from generated results
    for fragment in fragments:
        new_atoms = {"N":[], "CA":[], "C":[], "O":[]}
        offset = {"N":0, "CA":1, "C":2, "O":3}
        for atom in input_structure.atoms:
            vec_index = 3*(atom.residue_id - start) + offset[atom._atom_name]
            if atom.residue_id >= start and atom.residue_id <= end:
                if atom._atom_name == "O":
                    if atom.residue_id == end:
                        new_atoms["O"].append(MissingO(ss=atom.ss, id=atom.id, atom_name=atom._atom_name, residue=atom.residue, chain_name=atom.chain_name, residue_id=atom.residue_id, coordinates=atom.coordinates))
                        continue
                    ca = fragment[vec_index-2].numpy(force=True)
                    c = fragment[vec_index-1].numpy(force=True)
                    n = fragment[vec_index+1].numpy(force=True)
                    ca_c = c - ca
                    ca_c /= np.linalg.norm(ca_c)
                    n_c = c - n
                    n_c /= np.linalg.norm(n_c)
                    bisector = ca_c + n_c
                    bisector /= np.linalg.norm(bisector)
                    coordinates = c + bisector*1.23
                else:
                    coordinates = fragment[vec_index].numpy(force=True)
                new_atom = Atom(ss=atom.ss, id=atom.id, atom_name=atom._atom_name, residue=atom.residue, chain_name=atom.chain_name, residue_id=atom.residue_id, coordinates=coordinates)
                new_atoms[atom._atom_name].append(new_atom)
            else:
                new_atoms[atom._atom_name].append(atom)

        structure = Structure(atoms=(new_atoms["CA"], new_atoms["C"], new_atoms["N"], new_atoms["O"]), missing_o_index=end-1)
        new_structures.append(structure)


    new_structures.sort(key=lambda structure: torch.linalg.vector_norm(torch.tensor(structure.local_displacement(end,start-1)) - displacement[0]).item())
    disp = torch.linalg.vector_norm(fragments[:,-1,:].cpu()-prev_three[:,-1,:]-displacement, dim=1).numpy(force=True)
    plt.hist(disp, bins=100)
    if not os.path.exists(f"{os.path.dirname(__file__)}/plots"):    
        os.makedirs(f"{os.path.dirname(__file__)}/plots")
    plt.savefig(f"{os.path.dirname(__file__)}/plots/histogram.png")
    plt.close()
    pdb_name = os.path.splitext(os.path.basename(pdb))[0]
    if not os.path.exists(f"{os.path.dirname(__file__)}/generations"):
        os.makedirs(f"{os.path.dirname(__file__)}/generations")
    output_path = f"{os.path.dirname(__file__)}/generations/{pdb_name}_output.pdb"

    output_file = open(output_path, "w")

    for i, structure in enumerate(new_structures):
        print(f"MODEL {i+1}", file=output_file)

        lines = structure.to_pdb()
        for line in lines:
            print(line, file=output_file)
        print("ENDMDL", file=output_file)

    output_file.close()

    table = [["Amino acids sequence", f"{aa}"], ["Secondary structure", f"{ss}"]]
    print(tabulate(table))
