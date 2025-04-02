from sklearn.neighbors import KDTree
import numpy as np
from core.model_torch_3 import BertCVAE
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    baseline = np.load("validation/helix_14.npy")
    print(baseline.shape)
    baseline = baseline.reshape(baseline.shape[0], -1)
    model = BertCVAE(None, None, 6, 0.01, 0.01, 1e-3, 200, "cuda", None).cuda().eval()
    model.load_state_dict(torch.load("model_7.pt", weights_only=True))
    tree = KDTree(baseline)
    ss = torch.ones(1000, 15).cuda()
    aa = torch.randint(0, 20, (1000, 15)).cuda()
    displacement = torch.rand(1000, 3).cuda() * 10
    print(displacement)
    first_three = torch.stack([torch.tensor([-6.093,0.0,0.0]), torch.tensor([-4.065,0.0,0.0]), torch.tensor([-1.941,0.0,0.0])]).unsqueeze(0).expand(1000, -1, -1).cuda()
    labels = torch.stack([aa, ss], dim=2).long().cuda()
    with torch.no_grad():
        # print(first_three.shape, labels.shape, displacement.shape)
        x = model.generate(None, first_three, labels, displacement)
    x = x - x[:,1].unsqueeze(1)
    print(x)
    print(torch.linalg.vector_norm(x[:,-1,:]-displacement, dim=1))
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
    