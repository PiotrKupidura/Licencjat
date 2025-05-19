from typing import List
import numpy as np

# tools for reading PDB files
# functionalities are dedicated to parse alpha carbon trace including secondary structure

RESIDUES = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
RESIDUES_NUM = {"ALA": 1,  "ARG": 2,  "ASN": 3,  "ASP": 4,
            "CYS": 5,  "GLN": 6,  "GLU": 7,  "GLY": 8,
            "HIS": 9,  "ILE": 10, "LEU": 11, "LYS": 12,
            "MET": 13, "PHE": 14, "PRO": 15, "SER": 16,
            "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20}

STRUCTURES = {"H": 1, "E": 2, "C": 3}


class Atom:
    def __init__(self, ss, id, atom_name, residue, residue_id, chain_name, coordinates):
        self._ss = ss
        self._id = id
        self._atom_name = atom_name
        self._residue = residue
        self._residue_id = residue_id
        self._chain_name = chain_name
        self._coordinates = coordinates

    def __str__(self):
        formatters = ("ATOM", self._id, self._atom_name, self.residue, self.chain_name, self.residue_id, self.coordinates[0],
                      self.coordinates[1], self.coordinates[2], 1.00, 0.00, "C")
        return "%4s %6d %3s %4s %s %4d %10.3f %7.3f %7.3f %5.2f %5.2f %11s" % formatters

    def __lt__(self, other):
        return self.residue_id < other.residue_id

    def __gt__(self, other):
        return self.residue_id > other.residue_id

    @property
    def ss(self):
        return self._ss

    @property
    def id(self):
        return self._id

    @property
    def residue(self):
        return self._residue

    @property
    def residue_id(self):
        return self._residue_id

    @property
    def chain_name(self):
        return self._chain_name

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def z(self):
        return self.coordinates[2]

    @coordinates.setter
    def coordinates(self, vector):
        self._coordinates = vector


class Structure:
    def __init__(self, atoms: List[Atom]):
        self._ca, self._c, self._n = atoms
        self.atoms = []
        if len(self._n) != len(self._ca) or len(self._ca) != len(self._c):
            print(len(self._n), len(self._ca), len(self._c))
        for i in range(min(len(self._n), len(self._ca), len(self._c))):
            self.atoms.append(self._n[i])
            self.atoms.append(self._ca[i])
            self.atoms.append(self._c[i])

    def length(self):
        return len(self.atoms)

    def coordinates(self):
        # get list of coordinates of all atoms
        return self.coordinates

    def set_coordinates(self, coordinates):
        for i in range(self.length()):
            self.atoms[i].coordinates = coordinates[i]

    def find_residue(self, residue_id):
        for atom in self._ca:
            if atom.residue_id == residue_id:
                return self._ca.index(atom)
            else:
                continue

    def read_sequence(self, i, j):
        string = ""
        atoms = self._ca[self.find_residue(i):self.find_residue(j)+1]
        for atom in atoms:
            string += RESIDUES[atom.residue[-3:]]
        return string

    def read_secondary_structure(self, i, j):
        string = ""
        atoms = self._ca[self.find_residue(i):self.find_residue(j)+1]
        for atom in atoms:
            string += atom.ss
        return string

    def local_displacement(self, i, j):
        return self._c[self.find_residue(i)].coordinates - self._c[self.find_residue(j)].coordinates

    def split_chains(self):
        chains = {}
        for atom in self.atoms:
            if atom.chain_name in chains:
                chains[atom.chain_name].append(atom)
            else:
                chains[atom.chain_name] = [atom]
        for chain in chains:
            chains[chain] = Structure(atoms=chains[chain])
        return chains

    def generate_observations(self, len_fragment, weight):
        inputs = []
        labels = []
        for i in range(self.length() - len_fragment + 1):
            inp = np.zeros((len_fragment,3,3))
            lab = np.zeros((len_fragment,3))
            ignore = False
            for j, ca, c, n in zip(range(len_fragment), self._ca[i:i+len_fragment], self._c[i:i+len_fragment], self._n[i:i+len_fragment]):
                inp[j] = np.array([[n.x, n.y, n.z], [ca.x, ca.y, ca.z], [c.x, c.y, c.z]])
                if ca.residue in RESIDUES_NUM:
                    lab[j, 0] = RESIDUES_NUM[ca.residue]
                else:
                    ignore = True
                    break
                lab[j, 1] = STRUCTURES[ca.ss]
                lab[j, 2] = weight
            if ignore:
                ignore = False
                continue
            if self.atoms[i+len_fragment-1].residue_id - self.atoms[i].residue_id == len_fragment - 1:
                inputs.append(inp)
                labels.append(lab)
        if len(inputs) > 0:
            return np.stack(inputs, dtype=np.float32), np.stack(labels, dtype=np.float32)
        return None, None

    def to_pdb(self):
        return [atom.__str__() for atom in self.atoms]


class LineParser:
    def __init__(self, line):
        self._line = line

    @property
    def line(self):
        return self._line

    def parse_id(self):
        ORDINAL_START = 6
        ORDINAL_END = 11
        return int(self.line[ORDINAL_START:ORDINAL_END].strip())

    def parse_atom_name(self):
        ORDINAL_START = 13
        ORDINAL_END = 16
        return self.line[ORDINAL_START:ORDINAL_END].strip()

    def parse_residue(self):
        ORDINAL_START = 16
        ORDINAL_END = 20
        return self.line[ORDINAL_START:ORDINAL_END].strip()

    def parse_residue_id(self):
        ORDINAL_START = 22
        ORDINAL_END = 26
        return int(self.line[ORDINAL_START:ORDINAL_END].strip())

    def parse_chain_name(self):
        ORDINAL = 21
        return self.line[ORDINAL]

    def parse_x(self):
        ORDINAL_START = 30
        ORDINAL_END = 38
        return float(self.line[ORDINAL_START:ORDINAL_END].strip())

    def parse_y(self):
        ORDINAL_START = 38
        ORDINAL_END = 46
        return float(self.line[ORDINAL_START:ORDINAL_END].strip())

    def parse_z(self):
        ORDINAL_START = 46
        ORDINAL_END = 54
        return float(self.line[ORDINAL_START:ORDINAL_END].strip())


class FileParser:
    def __init__(self, file):
        self.file = file
        stream = open(file)
        self._lines = stream.readlines()
        stream.close()

    @property
    def lines(self):
        return [line for line in self._lines if len(line) >= 4]

    def parse_ca(self):
        records = []
        for line in self.lines:
            ORDINAL_START_ATOM = 0
            ORDINAL_END_ATOM = 4
            if line[ORDINAL_START_ATOM:ORDINAL_END_ATOM] == "ATOM":
                ORDINAL_START_CA = 12
                ORDINAL_END_CA = 16
                if line[ORDINAL_START_CA:ORDINAL_END_CA].strip() == "CA":
                    records.append(line)
        return records

    def parse_atoms(self):
        records = []
        for line in self.lines:
            ORDINAL_START_ATOM = 0
            ORDINAL_END_ATOM = 4
            if line[ORDINAL_START_ATOM:ORDINAL_END_ATOM] == "ATOM":
                ORDINAL_START_CA = 13
                ORDINAL_END_CA = 16
                # print(line[ORDINAL_START_CA:ORDINAL_END_CA].strip())
                if line[ORDINAL_START_CA:ORDINAL_END_CA].strip() in ["CA", "C", "N"]:
                    records.append(line)
        return records

    def parse_helix(self):
        numbers = []
        for line in self.lines:
            ORDINAL_START_HELIX = 0
            ORDINAL_END_HELIX = 5
            if line[ORDINAL_START_HELIX:ORDINAL_END_HELIX] == "HELIX":
                ORDINAL_START_INITIAL = 21
                ORDINAL_END_INITIAL = 25
                initial_residue_id = int(line[ORDINAL_START_INITIAL:ORDINAL_END_INITIAL].strip())
                ORDINAL_START_TERMINAL = 33
                ORDINAL_END_TERMINAL = 37
                terminal_residue_id = int(line[ORDINAL_START_TERMINAL:ORDINAL_END_TERMINAL].strip())
                for i in range(initial_residue_id, terminal_residue_id+1):
                    numbers.append(i)
        return numbers

    def parse_sheet(self):
        numbers = []
        for line in self.lines:
            ORDINAL_START_SHEET = 0
            ORDINAL_END_SHEET = 5
            if line[ORDINAL_START_SHEET:ORDINAL_END_SHEET] == "SHEET":
                ORDINAL_START_INITIAL = 22
                ORDINAL_END_INITIAL = 26
                initial_residue_id = int(line[ORDINAL_START_INITIAL:ORDINAL_END_INITIAL].strip())
                ORDINAL_START_TERMINAL = 33
                ORDINAL_END_TERMINAL = 37
                terminal_residue_id = int(line[ORDINAL_START_TERMINAL:ORDINAL_END_TERMINAL].strip())
                for i in range(initial_residue_id, terminal_residue_id+1):
                    numbers.append(i)
        return numbers

    def load_atoms(self):
        ca = []
        c = []
        n = []
        records = self.parse_atoms()
        if len(records) == 0:
            return ca, c, n
        parser = LineParser(records[0])
        chain = parser.parse_chain_name()
        for record in records:
            parser = LineParser(record)
            id = parser.parse_id()
            atom_name = parser.parse_atom_name()
            residue = parser.parse_residue()
            residue_id = parser.parse_residue_id()
            chain_name = parser.parse_chain_name()
            if chain_name != chain:
                continue
            x = parser.parse_x()
            y = parser.parse_y()
            z = parser.parse_z()
            coordinates = np.array([float(x), float(y), float(z)])
            # search secondary structure for each atom
            if residue_id not in self.parse_helix() and residue_id not in self.parse_sheet():
                ss = "C"
            else:
                if residue_id in self.parse_helix():
                    ss = "H"
                if residue_id in self.parse_sheet():
                    ss = "E"
            if len(residue) == 3 or residue[0] == "A":
                if atom_name == "CA":
                    ca.append(Atom(ss=ss, id=id, atom_name=atom_name, residue=residue, residue_id=residue_id, chain_name=chain_name, coordinates=coordinates))
                elif atom_name == "C":
                    c.append(Atom(ss=ss, id=id, atom_name=atom_name, residue=residue, residue_id=residue_id, chain_name=chain_name, coordinates=coordinates))
                elif atom_name == "N":
                    n.append(Atom(ss=ss, id=id, atom_name=atom_name, residue=residue, residue_id=residue_id, chain_name=chain_name, coordinates=coordinates))
        return ca, c, n

    def load_structure(self, chain=None):
        return Structure(atoms=self.load_atoms())
