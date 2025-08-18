from sklearn.neighbors import BallTree
import numpy as np
from core.model import CVAE
import torch
import matplotlib.pyplot as plt
import argparse
import json
from Bio.SVDSuperimposer import SVDSuperimposer
from sklearn.metrics import DistanceMetric

def parse_config(config):
    config_file = open(config, "r")
    parameters = json.loads(config_file.read())
    length = parameters["n"]
    latent_dim = parameters["latent_dim"]
    return length, latent_dim

def superimpose(x, y):
    sup = SVDSuperimposer()
    x = x.reshape(-1,3)
    y = y.reshape(-1,3)
    sup.set(x, y)
    sup.run()
    return sup.get_rms()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("-dir_read", type=str, required=True)
    parser.add_argument("-n", type=int, required=True)
    args = parser.parse_args()

    config = "config.json"
    length, latent_dim = parse_config(config)

    baseline = np.load(f"{args.dir_read}/helix_{length}.npy")
    baseline = baseline.reshape(baseline.shape[0], -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(length,latent_dim,0,0,0).to(device).eval()
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))

    metric = DistanceMetric.get_metric("pyfunc", func=superimpose)
    tree = BallTree(baseline, metric=metric)
    ss = torch.ones(args.n, length, device=device)
    # aa = torch.zeros(1000, 15, device=device) # unknown amino acid (works worse than random)
    aa = torch.randint(1, 21, (args.n, length), device=device) # random amino acids

    displacement = torch.rand(args.n, 3, device=device) * 10 # random displacement
    # example coordinates of the previous three atoms that obey the bond length and angle constraints
    first_three = torch.stack([torch.tensor([-2.156,1.131,0.0], device=device), 
                               torch.tensor([-1.458,0.0,0.0], device=device),
                               torch.tensor([0.0,0.0,0.0], device=device)]).unsqueeze(0).expand(args.n, -1, -1)
    labels = torch.stack([aa, ss], dim=2)

    with torch.no_grad():
        x = model.generate(args.n, first_three, aa, ss, displacement)

    x = x - x[:,1].unsqueeze(1) # center on the first c_alpha
    ind = torch.tensor(list(range(1, length*3, 3)), dtype=torch.long, device=device)
    x = x[:,ind,:] # extract c_alpha coordinates
    x = x.flatten(1).cpu().numpy()

    distance, neighbors = tree.query(x, k=1, return_distance=True)
    plt.hist(distance)
    plt.xlabel("RMSD")
    plt.ylabel("Count")
    plt.savefig(f"{args.dir_read}/histogram.png")
    plt.close()

    distance, neighbors = tree.query(baseline, k=2, return_distance=True)
    plt.hist(distance[:,1])
    plt.xlabel("RMSD")
    plt.ylabel("Count")
    plt.savefig(f"{args.dir_read}/histogram_baseline.png")
    plt.close()
