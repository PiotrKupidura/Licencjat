import torch
import math

"""Pytorch implementation of NeRF backbone reconstruction method from
   "Practical conversion from torsion space to Cartesian space for in silico protein synthesis".
   Based on https://github.com/sokrypton/tf_proteins/blob/master/coord_to_dihedrals_tools.ipynb"""

def to_ang(a,b,c):
    '''
    ====================================================================
    given coordinates a-b-c, return angle
    ====================================================================
    '''
    ba = b-a
    bc = b-c
    ang = torch.acos(torch.sum(ba*bc,-1)/(torch.norm(ba,dim=-1)*torch.norm(bc,dim=-1)+1e-8)+1e-8)
    return ang

def to_dih(a,b,c,d):
    '''
    ====================================================================
    given coordinates a-b-c-d, return dihedral
    ====================================================================
    '''
    bc = torch.nn.functional.normalize(b-c+1e-8,dim=-1)
    n1 = torch.linalg.cross(a-b,bc)
    n2 = torch.linalg.cross(bc,c-d)
    x = torch.sum(n1*n2,-1)
    y = torch.sum(torch.linalg.cross(n1,bc)*n2,-1)
    dih = torch.atan2(y,x+1e-8)
    return dih

def extend(a,b,c,rotation):
    '''
    =================================================================
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    =================================================================
    ref: https://doi.org/10.1002/jcc.20237
    =================================================================
    '''
    bc = torch.nn.functional.normalize(b-c+1e-8,dim=-1)
    n = torch.nn.functional.normalize(torch.linalg.cross(b-a, bc)+1e-8,dim=-1)
    m = torch.hstack([bc,torch.linalg.cross(n,bc),n])
    d = rotation.unsqueeze(1).transpose(1,2)
    res = c + (m*d).sum(1,keepdim=True)
    return res


class CartesianToDihedral(torch.nn.Module):
    def __init__(self):
        super(CartesianToDihedral, self).__init__()

    def forward(self, inputs, return_angles=False):
        """
        Input:
        inputs: cartesian coordinates (batch_size, num_res, 3, 3)
        Output:
        angles: dihedral angles (batch_size, num_res, 2)
        first_three: coordinates of the first three atoms (batch_size, 3, 3)
        """
        inputs = torch.stack(torch.split(inputs.flatten(1,2),inputs.shape[1]//3,dim=-1)).transpose(0,1).transpose(1,2).squeeze(2)
        angles = []
        angles_out = []
        for i in range(0, inputs.shape[1]-3):
            a,b,c,d = torch.split(inputs[:,i:i+4,:],1,dim=1)
            dih = to_dih(a,b,c,d)
            angles.append(torch.stack([torch.sin(dih), torch.cos(dih)]))

        angles = torch.stack(angles, dim=-1).squeeze([2,3]).transpose(0,1).transpose(1,2)
        first_three = inputs[:,:3,:]

        if return_angles:
            return angles, first_three, torch.cat(angles_out)

        angles = torch.cat([angles[:,:,i] for i in range(2)], dim=-1) #s1s2c1c2
        return angles, first_three.squeeze(2)

class DihedralToCartesian(torch.nn.Module):
    def __init__(self):
        super(DihedralToCartesian, self).__init__()

    def forward(self, inputs, return_angles=False, train_data=None) -> torch.Tensor:
        """
        Inputs:
        angles: dihedral angles (batch_size, num_res, 2)
        first_three: coordinates of the first three atoms (batch_size, 3, 3)
        Output:
        cartesian coordinates (batch_size, num_res*3, 3)
        """
        angles, prev_three = inputs
        angles = torch.reshape(torch.cat(torch.chunk(angles,2,dim=-1),dim=1),(angles.shape[0],2,angles.shape[1]//2)).transpose(1,2) #s1s2c1c2

        res = torch.zeros((angles.shape[0], angles.shape[1], 3), device=angles.device)
        a, b, c = torch.split(prev_three,1,dim=1)

        sin_theta, cos_theta = torch.split(angles,1,dim=2)
        n_theta = (sin_theta**2 + cos_theta**2 + 1e-8).sqrt()
        sin_theta, cos_theta = sin_theta/n_theta, cos_theta/n_theta

        alpha = torch.tensor([2.028, 2.124, 1.941]*(angles.shape[1]//3), device=angles.device).unsqueeze(0).expand(angles.shape[0],-1)
        sin_alpha = torch.sin(alpha)
        cos_alpha = torch.cos(alpha)

        # Each atoms rotation does not depend on its position and thus can be computed once
        ca_c_bond, c_n_bond, n_ca_bond = 1.458, 1.523, 1.329
        rotation = torch.stack([cos_alpha, sin_alpha*cos_theta.squeeze(2), -sin_alpha*sin_theta.squeeze(2)], dim=2) \
        * torch.tensor([n_ca_bond, ca_c_bond, c_n_bond]*(angles.shape[1]//3), device=angles.device).unsqueeze(0).unsqueeze(2).expand(angles.shape[0],-1,3)

        for i in range(0, angles.shape[1], 1):
            d_cart = extend(a,b,c,rotation[:,i,:])
            if train_data is not None:
                a, b, c = b, c, train_data[:,i,:].unsqueeze(1)
            else:
                a, b, c = b, c, d_cart

            res[:,i,:] = d_cart.squeeze(1)

        if return_angles:
            res_angles = torch.atan2(sin_theta.squeeze(2), cos_theta.squeeze(2)+1e-8).unflatten(1, (14,3)).flatten(0,1)
            return res, res_angles

        return res
