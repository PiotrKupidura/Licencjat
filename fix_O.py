import math

def parse_atom_line(line):
#  try:
    return {
        'record': line[0:6].strip(),
        'atom_serial': int(line[6:11]),
        'atom_name': line[12:16].strip(),
        'res_name': line[17:20].strip(),
        'chain_id': line[21],
        'res_seq': int(line[22:26]),
        'x': float(line[30:38]),
        'y': float(line[38:46]),
        'z': float(line[46:54]),
        'line': line
    }
#  except:
#    print(f"Problematic line:\n{line}")

def format_atom_line(atom_dict, atom_serial, atom_name, x, y, z):
    return f"ATOM  {atom_serial:5d} {atom_name:^4} {atom_dict['res_name']:>3} {atom_dict['chain_id']}{atom_dict['res_seq']:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O  "

def vector_subtract(a, b):
    return [a[i] - b[i] for i in range(3)]

def vector_add(a, b):
    return [a[i] + b[i] for i in range(3)]

def vector_scale(v, s):
    return [v[i] * s for i in range(3)]

def normalize(v):
    norm = math.sqrt(sum([x**2 for x in v]))
    return [x / norm for x in v]

def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

def reconstruct_O(CA, C, N_next):
    # Estimate O position in a tetrahedral geometry relative to N, CA, and C
    CA_C = normalize(vector_subtract(C, CA))
    N_C = normalize(vector_subtract(C, N_next))

    bisector = normalize(vector_add(CA_C, N_C))
    O = vector_add(C, vector_scale(bisector, 1.23))  # C=O bond length ~1.23 Å

    return O

def process_pdb(pdb_lines):
    models = []
    current_model = []
    for line in pdb_lines:
        if line.startswith("MODEL"):
            current_model = []
        elif line.startswith("ENDMDL"):
            models.append(current_model)
        elif line.startswith("ATOM"):
            current_model.append(parse_atom_line(line))
    
    output_lines = []
    for model_index, model in enumerate(models):
        output_lines.append(f"MODEL     {model_index + 1}")
        atom_serial = 4
        
        for i in range(len(model)):
            atom = model[i]
            l = atom['line'].rstrip()
            if atom["atom_name"][0] == "N":
                l = l[0:-1]+"N"
            output_lines.append(l)

            # Add O if this is a C atom and next residue exists
            if atom['atom_name'] == 'C':
                # Get N and CA of the same residue
                ca_atom = next((a for a in model if a['res_seq'] == atom['res_seq'] and a['atom_name'] == 'CA'), None)
                n_next_atom = next((a for a in model if a['res_seq'] == atom['res_seq'] + 1 and a['atom_name'] == 'N'), None)

                if n_next_atom and ca_atom:
                    N = [n_next_atom['x'], n_next_atom['y'], n_next_atom['z']]
                    CA = [ca_atom['x'], ca_atom['y'], ca_atom['z']]
                    C = [atom['x'], atom['y'], atom['z']]
                    
                    O = reconstruct_O(CA, C, N)
                    o_line = format_atom_line(atom, atom_serial, ' O  ', *O)
                    output_lines.append(o_line)
                    atom_serial += 10
        output_lines.append("ENDMDL")
    return output_lines

# Main script
if __name__ == "__main__":
    with open("input.pdb", "r") as f:
        pdb_lines = f.readlines()
    
    new_lines = process_pdb(pdb_lines)
    
    with open("output_with_O.pdb", "w") as f:
        for line in new_lines:
            f.write(line + "\n")

