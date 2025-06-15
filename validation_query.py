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
    model = CVAE(length,latent_dim,0,0,0).cuda().eval()
    model.load_state_dict(torch.load(args.model, weights_only=True))
    metric = DistanceMetric.get_metric("pyfunc", func=superimpose)
    tree = BallTree(baseline, metric=metric)
    ss = torch.ones(args.n, length).cuda()
    # aa = torch.zeros(1000, 15).cuda() # unknown amino acid (works worse than random)
    aa = torch.randint(1, 21, (args.n, length)).cuda() # random amino acids
    displacement = torch.rand(args.n, 3).cuda() * 10
    first_three = torch.stack([torch.tensor([-6.093,0.0,0.0]), torch.tensor([-4.065,0.0,0.0]), torch.tensor([-1.941,0.0,0.0])]).unsqueeze(0).expand(args.n, -1, -1).cuda()
    labels = torch.stack([aa, ss], dim=2).long().cuda()
    with torch.no_grad():
        x = model.generate(args.n, first_three, aa, ss, displacement)
    x = x - x[:,1].unsqueeze(1)
    ind = torch.tensor(list(range(1, length*3, 3)), dtype=torch.long).cuda()
    x = x[:,ind,:]
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
