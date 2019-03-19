import numpy as np
import torch

def l1shrink(eps, tensor, device):
    result = torch.zeros((tensor.size()[0], tensor.size()[1])).to(device)
    result = result + eps*(tensor<-eps).float()
    result = result - eps*(tensor>eps).float()
    return result

def l21shrink(eps, tensor, device):
    result = torch.zeros((tensor.size()[0], tensor.size()[1])).to(device)
    norms = torch.norm(tensor, dim=0)
    for j in range(tensor.size()[1]):
        if norms[j] > eps:
            result[:, j] = tensor[:, j] - eps * tensor[:, j] / norms[j]
        else:
            result[:, j] = 0.
    return result


# def shrink(epsilon, x):
#     """
#     @Original Author: Prof. Randy
#     @Modified by: Chong Zhou
#     Args:
#         epsilon: the shrinkage parameter (either a scalar or a vector)
#         x: the vector to shrink on
#     Returns:
#         The shrunk vector
#     """
#     output = torch.zeros((x.size()[0], x.size()[1]))
# #     output = np.array(x*0.)
    
#     for i in range(x.size()[0]):
#         if x[i] > epsilon:
            
#     for i in range(x.size()[0]):
#         for j in range(x.size()[1]):
#             if x[i,j] > epsilon:
#                 output[i,j] = x[i,j] - epsilon
#             elif x[i,j] < -epsilon:
#                 output[i,j] = x[i,j] + epsilon
#             else:
#                 output[i,j] = 0
#     return output