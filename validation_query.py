from re import L
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
    # print(x.shape, y.shape)
    sup.set(x, y)
    sup.run()
    return sup.get_rms()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    length, latent_dim = parse_config(args.config)
    baseline = np.load(f"validation/helix_{length}.npy")
    baseline = baseline.reshape(baseline.shape[0], -1)
    model = CVAE(length,latent_dim,0,0,0).cuda().eval()
    model.load_state_dict(torch.load(args.model, weights_only=True))
    metric = DistanceMetric.get_metric("pyfunc", func=superimpose)
    tree = BallTree(baseline, metric=metric)
    ss = torch.ones(1000, 15).cuda()
    # aa = torch.zeros(1000, 15).cuda()
    aa = torch.randint(1, 21, (1000, 15)).cuda()
    displacement = torch.rand(1000, 3).cuda() * 10
    first_three = torch.stack([torch.tensor([-6.093,0.0,0.0]), torch.tensor([-4.065,0.0,0.0]), torch.tensor([-1.941,0.0,0.0])]).unsqueeze(0).expand(1000, -1, -1).cuda()
    labels = torch.stack([aa, ss], dim=2).long().cuda()
    with torch.no_grad():
        x = model.generate(1000, first_three, labels, displacement)
    x = x - x[:,1].unsqueeze(1)
    ind = torch.tensor(list(range(1, 42, 3)), dtype=torch.long).cuda()
    x = x[:,ind,:]
    x = x.flatten(1).cpu().numpy()
    distance, neighbors = tree.query(x, k=1, return_distance=True)
    plt.hist(distance)
    plt.savefig("validation/histogram.png")
    plt.close()
    distance, neighbors = tree.query(baseline, k=2, return_distance=True)
    plt.hist(distance[:,1])
    plt.savefig("validation/histogram_baseline.png")
    plt.close()
