import numpy as np
from core.model import CVAE
import torch
import matplotlib.pyplot as plt
import argparse
import json
import miniball

def parse_config(config):
    config_file = open(config, "r")
    parameters = json.loads(config_file.read())
    length = parameters["n"]
    latent_dim = parameters["latent_dim"]
    return length, latent_dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("-dir_read", type=str, required=True)
    parser.add_argument("-n", type=int, required=True)
    args = parser.parse_args()
    config = "config.json"
    length, latent_dim = parse_config(config)
    model = CVAE(length,latent_dim,0,0,0).cuda().eval()
    model.load_state_dict(torch.load(args.model, weights_only=True))
    ss = torch.ones(args.n, length).cuda()
    # aa = torch.zeros(1000, 15).cuda() # unknown amino acid (works worse than random)
    aa = torch.randint(1, 21, (args.n, length)).cuda() # random amino acids
    displacement = torch.rand(1, 3).cuda().repeat(args.n, 1) * length
    first_three = torch.stack([torch.tensor([-2.156,1.131,0.0]), torch.tensor([-1.458,0.0,0.0]), torch.tensor([0.0,0.0,0.0])]).unsqueeze(0).expand(args.n, -1, -1).cuda()
    labels = torch.stack([aa, ss], dim=2).long().cuda()
    with torch.no_grad():
        x = model.generate(args.n, first_three, aa, ss, displacement)
    # x = x - x[:,1].unsqueeze(1)
    gen_displacement = x[:,-1] - first_three[:,-1] - displacement
    x = x[gen_displacement.norm(dim=1) < 1]
    radii = []
    for i in range(length*3):
        c, r = miniball.get_bounding_ball(x[:,i].cpu().numpy())
        print(i,r)
        radii.append(r)
    plt.plot(np.array(radii))
    plt.xlabel("Atom")
    plt.ylabel("Radius (Å)")
    plt.savefig(f"{args.dir_read}/radii.png")
    plt.close()
