import numpy as np
from core.model import CVAE
import torch
import matplotlib.pyplot as plt
import argparse
import json
import miniball
import os

def parse_config(config):
    config_file = open(config, "r")
    parameters = json.loads(config_file.read())
    length = parameters["n"]
    latent_dim = parameters["latent_dim"]
    return length, latent_dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("-n", type=int, required=True)
    args = parser.parse_args()

    config = "config.json"
    length, latent_dim = parse_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(length,latent_dim,0,0,0).to(device).eval()
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))

    ss = torch.ones(args.n, length, device=device)
    # aa = torch.zeros(1000, 15, device=device) # unknown amino acid (works worse than random)
    aa = torch.randint(1, 21, (args.n, length), device=device) # random amino acids

    displacement = torch.rand(1, 3, device=device).expand(args.n, -1) * 10 # random displacement
    # example coordinates of the previous three atoms that obey the bond length and angle constraints
    first_three = torch.stack([torch.tensor([-2.156,1.131,0.0], device=device), 
                               torch.tensor([-1.458,0.0,0.0], device=device),
                               torch.tensor([0.0,0.0,0.0], device=device)]).unsqueeze(0).expand(args.n, -1, -1)
    labels = torch.stack([aa, ss], dim=2)

    with torch.no_grad():
        x = model.generate(args.n, first_three, aa, ss, displacement)

    gen_displacement = x[:,-1] - displacement
    x = x[gen_displacement.norm(dim=1) < 1] # only include closely fitting fragments

    radii = []
    for i in range(length*3):
        c, r = miniball.get_bounding_ball(x[:,i].cpu().numpy())
        print(i,r)
        radii.append(r)
    plt.plot(np.array(radii))
    plt.xlabel("Atom")
    plt.ylabel("Radius (Å)")
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/radii.png")
    plt.close()
