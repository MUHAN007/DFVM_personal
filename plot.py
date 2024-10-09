import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from GenerateData import *
from DFVM_Poisson_Singularity import MLP
from DFVM_Poisson_HighDim import Resnet
DIMENSION = 2
eps = 1e-3
N = 100
DIM_INPUT = DIMENSION  # Input dimension
DIM_OUTPUT = 1 

file_path = "D:\\workspace\\code_workspace\\DFVM\\PossionEQ_seed0\\2DIM-DFVM_net"
# DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
lb = [0 for _ in range(DIMENSION)]
ub = [1 for _ in range(DIMENSION)]


x_test =torch.Tensor(sampleCube(dim=2,l_bounds=lb,u_bounds=ub,N=N)).requires_grad_(True).to(DEVICE)
# *****************************************************************
##singularity
model = MLP(DIM_INPUT, DIM_OUTPUT, hidden_width=40)
#******************************************************************
# ##highdim
# model = Resnet(DIM_INPUT, 128, DIM_OUTPUT, 3).to(DEVICE)
# #******************************************************************
model.load_state_dict( torch.load(file_path,map_location=DEVICE))

u_test= model(x_test)[:,0]
x_ = x_test.cpu().detach().numpy()
u_= u_test.cpu().detach().numpy()

# x = torch.sin(torch.pi*x_test/2)
# u_ = torch.sum(x,1).cpu().detach().numpy()  
# u_ = torch.tanh(u).cpu().detach().numpy()
print(x_[0])
print(u_[0])

##绘图
x = [a[0] for a in x_]
y = [a[1] for a in x_]
z = u_.reshape(-1)
# x = x[::2]
# y = y[::2]
# z = z[::2]
X, Y = np.meshgrid(x, y)
points = np.vstack((x, y)).T
values = z

# # 使用griddata进行插值
# Z = griddata(points, values, (X, Y), method='cubic')
# fig = plt.figure()
# ax = fig.subplots()
# cp = ax.contourf(X,Y,Z, 10, cmap='viridis')
# fig.colorbar(cp)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title("result")

# plt.show()


# 输出dat文件
with open('.\DFVM\output\outfile_biharmonic_predict_double.dat','w+') as outputfile:
    outputfile.write("VARIABLES= \"X\",\" Y\",\" U\"\n")
    outputfile.write("ZONE I="+str(N)+" J="+str(N)+" K=1 f=point\n")
    for i in range(len(x)):
        outputfile.write(str(x[i])+" ")
        outputfile.write(str(y[i])+" ")
        outputfile.write(str(z[i])+'\n')
