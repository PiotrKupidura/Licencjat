import os
import argparse
import logging
import numpy as np
# import tensorflow as tf
import torch
from tabulate import tabulate
# from core.model_torch_3 import DecoderLoader
from core.model_torch_3 import Trainer
from core.features import LabelMLP
from core.parser import FileParser, Structure, Atom
# from utils import Vec3
import matplotlib.pyplot as plt

# logging.getLogger("tensorflow").disabled=True
# logging.getLogger("h5py._conv").disabled=True

BOND_LENGTH = 3.8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-aa", type=str, help="amino acids sequence")
    parser.add_argument("-ss", type=str, help="secondary structure")
    parser.add_argument("-f", "--file", type=str, help="PDB file")
    parser.add_argument("-s", "--start", type=int, help="initial residue")
    parser.add_argument("-e", "--end", type=int, help="terminal residue")
    parser.add_argument("-m", "--model", type=str, help="model to be used")
    parser.add_argument("-r", "--repeats", type=int, help="number of returned fragments")
    parser.add_argument("-p", "--population", type=int, help="number of fragments generated to choose the best one")
    args = parser.parse_args()

    pdb = args.file
    start = args.start
    end = args.end
    model = args.model
    repeats = args.repeats
    population = args.population

    input_structure = FileParser(file=pdb).load_structure()

    if args.aa == None:
        aa = input_structure.read_sequence(start-1, end)
    else:
        aa = args.aa

    if args.ss == None:
        ss = input_structure.read_secondary_structure(start-1, end)
    else:
        ss = args.ss

    displacement = input_structure.local_displacement(start-1, end)

    print(displacement)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = Trainer(config="config.json")
    t.load_model(model)
    labels = [LabelMLP(aa=aa, ss=ss, dx=displacement[0], dy=displacement[1], dz=displacement[2]).format() for _ in range(population)]
    labels = torch.stack(labels).squeeze(1)
    displacement_label = labels[:,-3:].float()
    labels = torch.stack(torch.chunk(labels[:, :-3], 25, dim=1), dim=1).transpose(1,2)
    labels = torch.cat([labels[:,:,:21].argmax(dim=2, keepdim=True), labels[:,:,21:].argmax(dim=2, keepdim=True)], dim=2) # one hot
    labels = torch.cat((labels, displacement_label.unsqueeze(1).expand(-1, labels.shape[1], -1)), dim=-1).float()

    # vectors = decoder.predict(labels) # raw data from decoder
    # outputs = [Output(vector) for vector in vectors]

    # bound atoms not included in rebuilt fragment
    c_1 = input_structure._n[input_structure.find_residue(start-1)].coordinates
    c_2 = input_structure._ca[input_structure.find_residue(start-1)].coordinates
    c_3 = input_structure._c[input_structure.find_residue(start-1)].coordinates

    displacement = torch.tensor(displacement).float().unsqueeze(0).expand(population,-1)
    prev_three = torch.stack([torch.tensor(c_1),torch.tensor(c_2),torch.tensor(c_3)]).unsqueeze(0).expand(population,-1,-1).float()
    fragments = t.model.generate(population, prev_three.to(device), labels.to(device), displacement)

    new_structures = [] # all structures obtained from generated results
    for fragment in fragments:
        new_atoms = {"N":[], "CA":[], "C":[]}
        offset = {"N":0, "CA":1, "C":2}
        for atom in input_structure.atoms:
            if atom.residue_id >= start and atom.residue_id <= end:
                vec_index = 3*(atom.residue_id - start) + offset[atom._atom_name]
                coordinates = fragment[vec_index].numpy(force=True)
                new_atom = Atom(ss=atom.ss, id=atom.id, atom_name=atom._atom_name, residue=atom.residue, chain_name=atom.chain_name, residue_id=atom.residue_id, coordinates=coordinates)
                new_atoms[atom._atom_name].append(new_atom)
            else:
                new_atoms[atom._atom_name].append(atom)

        structure = Structure(atoms=(new_atoms["CA"], new_atoms["C"], new_atoms["N"]))
        new_structures.append(structure)


    new_structures.sort(key=lambda structure: torch.nn.functional.mse_loss(torch.tensor(structure.local_displacement(start-1,end)), displacement_label, reduction="mean").item())
    print([torch.linalg.vector_norm(torch.tensor(structure.local_displacement(start,end)) - displacement_label[0]) for structure in new_structures[:10]])
    disp = torch.linalg.vector_norm(fragments[:,-1,:].cpu()-prev_three[:,-1,:]-displacement_label, dim=1).numpy(force=True)
    plt.hist(disp, bins=100)
    plt.savefig("plots/histogram.png")
    plt.close()
    pdb_name = os.path.splitext(os.path.basename(pdb))[0]
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
