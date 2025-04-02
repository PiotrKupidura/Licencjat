import torch
import math

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

# def extend(a,b,c, sin_alpha, cos_alpha, sin_theta, cos_theta):
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


class GradNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output.clone()
        # grad_x = torch.sign(grad_x) * grad_x.abs().exp()
        print(grad_x[0])
        # grad_x = grad_x/grad_x.mean()
        # grad_exp = grad_x.exp()
        # grad_x = grad_exp/grad_exp.mean() * grad_x.mean()
        # print(grad_x[0])
        return grad_x #* torch.arange(42, device=grad_x.device).unsqueeze(0).unsqueeze(2)


class CartesianToDihedral(torch.nn.Module):
    def __init__(self):
        super(CartesianToDihedral, self).__init__()

    def forward(self, inputs, return_angles=False):
        """
        input_dim = (batch_size, num_res, 3, 3)
        output_dim = ((batch_size, num_res, 3), (batch_size, 3, 3))
        """
        inputs = torch.stack(torch.split(inputs.flatten(1,2),inputs.shape[1]//3,dim=-1)).transpose(0,1).transpose(1,2).squeeze(2)
        # print(inputs[0,3:,:])
        angles = []
        angles_out = []
        for i in range(0, inputs.shape[1]-3):
            a,b,c,d = torch.split(inputs[:,i:i+4,:],1,dim=1)

            # alpha = to_ang(b,c,d)
            dih = to_dih(a,b,c,d)

            # if return_angles:
            #     angles_out.append(torch.cat([alpha,theta],dim=1))
  
            angles.append(torch.stack([torch.sin(dih), torch.cos(dih)]))
            # angles.append(dih)
        
        angles = torch.stack(angles, dim=-1).squeeze([2,3]).transpose(0,1).transpose(1,2)
        # angles = torch.stack(angles, dim=-1).squeeze(1)#.unsqueeze(2)
        # print(angles[0])

        # print(angles.shape)
        first_three = inputs[:,:3,:]

        if return_angles:
            return angles, first_three, torch.cat(angles_out)
        
        angles = torch.cat([angles[:,:,i] for i in range(2)], dim=-1) #a1a2t1t2
        # angles = torch.cat([angles[:,i,:] for i in range(inputs.shape[1]-3)], dim=-1) #a1t1a2t2
        return angles, first_three.squeeze(2)
    
class DihedralToCartesian(torch.nn.Module):
    def __init__(self):
        super(DihedralToCartesian, self).__init__()

    def forward(self, inputs, return_angles=False, train_data=None) -> torch.Tensor:
        """
        input_dim = ((batch_size, num_atoms, 3), (batch_size, 3, 3))
        output_dim = (batch_size, num_atoms, 3)
        """
        angles, prev_three = inputs
        # angles = torch.stack(torch.split(angles,3,dim=-1)).transpose(0,1).squeeze(dim=2) #a1t1a2t2
        angles = torch.reshape(torch.cat(torch.chunk(angles,2,dim=-1),dim=1),(angles.shape[0],2,angles.shape[1]//2)).transpose(1,2) #a1a2t1t2
        
        # angles = GradNorm().apply(angles)
        
        res = torch.zeros((angles.shape[0], angles.shape[1], 3), device=angles.device)
        a, b, c = torch.split(prev_three,1,dim=1)
        
        # sin_alpha, cos_alpha, sin_theta, cos_theta = torch.split(angles,1,dim=2)
        # n_alpha = (sin_alpha**2 + cos_alpha**2 + 1e-8).sqrt()
        # n_theta = (sin_theta**2 + cos_theta**2 + 1e-8).sqrt()
        # sin_alpha, cos_alpha, sin_theta, cos_theta = sin_alpha/n_alpha, cos_alpha/n_alpha, sin_theta/n_theta, cos_theta/n_theta
        # alpha, theta = torch.split(angles,1,dim=2)
        # sin_alpha, cos_alpha = torch.sin(alpha), torch.cos(alpha)
        # sin_theta, cos_theta = torch.sin(angles), torch.cos(angles)
        
        sin_theta, cos_theta = torch.split(angles,1,dim=2)
        n_theta = (sin_theta**2 + cos_theta**2 + 1e-8).sqrt()#.detach()
        # sin_theta, cos_theta = GradNorm().apply(sin_theta/n_theta), GradNorm().apply(cos_theta/n_theta)
        sin_theta, cos_theta = sin_theta/n_theta, cos_theta/n_theta
        
        alpha = torch.tensor([2.028, 2.124, 1.941]*(angles.shape[1]//3), device=angles.device).unsqueeze(0).expand(angles.shape[0],-1)
        # print(alpha.shape, angles.shape)
        sin_alpha = torch.sin(alpha)
        cos_alpha = torch.cos(alpha)
        # print(angles[0])
        # print(sin_theta[0], cos_theta[0])
        
        # print(sin_alpha.shape, cos_alpha.shape, sin_theta.shape, cos_theta.shape)
        
        # Each atoms rotation does not depend on its position and thus can be computed once
        # print(torch.tensor([1.329, 1.458, 1.523]*(angles.shape[1]//3), device=angles.device).unsqueeze(0).unsqueeze(2).expand(angles.shape[0],-1, 3))
        # rotation = torch.stack([cos_alpha, sin_alpha*cos_theta, -sin_alpha*sin_theta], dim=2) * torch.tensor([1.329, 1.458, 1.523]*(angles.shape[1]//3), device=angles.device).unsqueeze(0).unsqueeze(2).expand(angles.shape[0],-1,3)
        rotation = torch.stack([cos_alpha, sin_alpha*cos_theta.squeeze(2), -sin_alpha*sin_theta.squeeze(2)], dim=2) * torch.tensor([1.329, 1.458, 1.523]*(angles.shape[1]//3), device=angles.device).unsqueeze(0).unsqueeze(2).expand(angles.shape[0],-1,3)

        
        # if return_angles:
        #     # alpha = torch.atan2(sin_alpha, cos_alpha+1e-8)
        #     # theta = torch.atan2(sin_theta, cos_theta+1e-8)
        #     res_angles = torch.stack([alpha.flatten(),theta.flatten()], dim=1)
        
        # rotation = GradNorm().apply(rotation)

        for i in range(0, angles.shape[1], 1):
            d_cart = extend(a,b,c,rotation[:,i,:])
            if train_data is not None:
                a, b, c = b, c, train_data[:,i,:].unsqueeze(1)
            else:
                a, b, c = b, c, d_cart
            
            res[:,i,:] = d_cart.squeeze(1)

        if return_angles:
            # print(res[0])
            # print(sin_theta[0], cos_theta[0])
            res_angles = torch.atan2(sin_theta.squeeze(2), cos_theta.squeeze(2)+1e-8).unflatten(1, (14,3)).flatten(0,1)
            # res_angles = angles.squeeze(2).unflatten(1, (14,3)).flatten(0,1)
            # res_angles = angles.unflatten(1, (14,3)).flatten(0,1)
            # return res, res_angles
            return res, res_angles
        
        return res