# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import *
import os
import argparse
from GenerateData import *


# Parser
parser = argparse.ArgumentParser(description='DFVM')
parser.add_argument('--dimension', type=int, default=2, metavar='N',
                    help='dimension of the problem (default: 100)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
seed = parser.parse_args().seed
# Omega
DIMENSION = parser.parse_args().dimension
a = [-1 for _ in range(DIMENSION)]
b = [ 1 for _ in range(DIMENSION)]
# Finite Volume
EPSILON = 1e-5       # Domain size
BDSIZE = 1           # Boundary size: J_{\partial V} = 2^BDSIZE - 1

# Network
DIM_INPUT = DIMENSION  # Input dimension
NUM_UNIT = 40          # Number of neurons in a single layer
DIM_OUTPUT = 2         # Output dimension,output is u, v
NUM_LAYERS = 6         # Number of layers in the model

# Optimizer
IS_DECAY = 1
LEARN_RATE = 1e-3          # Learning rate
LEARN_FREQUENCY = 50       # Learning rate decay interval
LEARN_LOWWER_BOUND = 1e-4  # Lower bound of learning rate
LEARN_DECAY_RATE = 0.99    # Learning rate decay rate
LOSS_FN = nn.MSELoss()     # Loss function

# Training
CUDA_ORDER = "0"
NUM_TRAIN_SAMPLE = 2000           # Size of the training set
NUM_BOUND_SAMPLE = 2000 // DIMENSION
NUM_TRAIN_TIMES = 1               # Number of training samples
NUM_ITERATION = 10000             # Number of training iterations per sample

# Re-sampling
IS_RESAMPLE = 0
SAMPLE_FREQUENCY = 2000     # Re-sampling interval

# Testing
NUM_TEST_SAMPLE = 10000     # Number of test samples
TEST_FREQUENCY = 1          # Output interval

# Loss weight
BETA = 1000                 # Weight of the boundary loss function

# Save model
IS_SAVE_MODEL = 1           # Flag to save the model


class PoissonEQuation(object):
    def __init__(self, dimension, epsilon, bd_size, device):
        self.D      = dimension
        self.E      = epsilon
        self.B      = bd_size
        self.bdsize = 2**bd_size - 1
        self.device = device
        self.epsilon = 0.1

    def f1(self, X, model ):

        u0 = torch.min(torch.sqrt((X[:,0]+0.3).pow(2)+X[:,1].pow(2))
                        -0.3,torch.sqrt((X[:,0]-0.3).pow(2)+X[:,1].pow(2))-0.25)
        u = torch.tanh(u0/(torch.sqrt(torch.tensor(2))*self.epsilon))
        v = model(X)[:,1]
        f = (-v+u.pow(3)-u)/(self.epsilon*self.epsilon)
        return f.detach()

    def g1(self, X):
        u0 = torch.min(torch.sqrt((X[:,0]+0.3).pow(2)+X[:,1].pow(2))
                        -0.3,torch.sqrt((X[:,0]-0.3).pow(2)+X[:,1].pow(2))-0.25)
        u = torch.tanh(u0/(torch.sqrt(torch.tensor(2))*self.epsilon))
        g = u
        return g.detach()
    
    def f2(self, X):
        f = 0 * torch.ones(len(X), 1).to(self.device)
        return f.detach()

    def g2(self, X):
        g = 0 * torch.ones(len(X), 1).to(self.device)
        return g.detach()

    def u_exact(self, X):
        x = torch.sum(X,1)/self.D
        u = x.pow(2)+torch.sin(x)
        return u.detach()

    # sample the interior of the domain: Monte-Carlo or Quasi-Monte Carlo
    def interior(self, N=100):
        eps = self.E # np.spacing(1)
        l_bounds = [l+eps for l in a]
        u_bounds = [u-eps for u in b]
        X = torch.FloatTensor( sampleCubeMC(self.D, l_bounds, u_bounds, N) )
        # X = torch.FloatTensor( sampleCubeQMC(self.D, l_bounds, u_bounds, N) )
        return X.requires_grad_(True).to(self.device)

    # sample the boundary of the domain
    def boundary(self, n=100):
        x_boundary = []
        for i in range( self.D ):
            x = np.random.uniform(a[i], b[i], [2*n, self.D]) 
            x[:n,i] = b[i]
            x[n:,i] = a[i]
            x_boundary.append(x)
        x_boundary = np.concatenate(x_boundary, axis=0)
        x_boundary = torch.FloatTensor(x_boundary).requires_grad_(True).to(self.device)
        return x_boundary


    # sample a neighborhood of x with size
    def neighborhood(self, x, size):
        l_bounds = [t-self.E for t in x.cpu().detach().numpy()]
        u_bounds = [t+self.E for t in x.cpu().detach().numpy()]
        sample   = sampleCubeQMC(self.D, l_bounds, u_bounds, size)
        sample   = torch.FloatTensor( sample ).to(self.device)
        return sample

    def neighborhoodBD(self, X):
        lb = [-1 for _ in range(self.D-1)]
        ub = [ 1 for _ in range(self.D-1)]
        x_QMC   = sampleCubeQMC(self.D-1, lb, ub, self.B)
        x_nbound = []
        for i in range( self.D ):
            x_nbound.append( np.insert(x_QMC, i, [ 1], axis=1) )
            x_nbound.append( np.insert(x_QMC, i, [-1], axis=1) )
        x_nbound = np.concatenate(x_nbound, axis=0).reshape(1, -1, self.D)
        x_nbound = torch.FloatTensor(x_nbound).to(self.device)
        X = torch.unsqueeze(X, dim=1)
        X = X.expand(-1, x_nbound.shape[1], x_nbound.shape[2])
        X_bound = X + self.E*x_nbound
        X_bound = X_bound.reshape(-1, self.D)
        return X_bound.detach().requires_grad_(True)

    def outerNormalVec(self):
        bd_dir = torch.zeros(2*self.D*self.bdsize, self.D)
        for i in range( self.D ):
            bd_dir[    2*i*self.bdsize : (2*i+1)*self.bdsize, i] =  1
            bd_dir[(2*i+1)*self.bdsize : 2*(i+1)*self.bdsize, i] = -1
        bd_dir = bd_dir.reshape(1,-1)
        return bd_dir.detach().requires_grad_(True).to(self.device)


class DFVMsolver(object):
    def __init__(self, Equation, model, device):
        self.Eq     = Equation
        self.model  = model
        self.device = device   

    # calculate the gradient of u_theta
    def Nu1(self, X):
        u = self.model(X)[:,0]
        Du = torch.autograd.grad(outputs     = [u], 
                                inputs       = [X], 
                                grad_outputs = torch.ones_like(u),
                                allow_unused = True,
                                retain_graph = True,
                                create_graph = True)[0]
        return Du

    def Nu2(self, X):
        v = self.model(X)[:,1]
        Dv = torch.autograd.grad(outputs     = [v], 
                                inputs       = [X], 
                                grad_outputs = torch.ones_like(v),
                                allow_unused = True,
                                retain_graph = True,
                                create_graph = True)[0]
        return Dv
    # calculate the integral of $\nabla u_theta \cdot \vec{n}$ in the neighborhood of x
    def integrate_BD1(self, X, x_bd, bd_dir):
        n = len(X)
        integrate_bd = torch.zeros(n, 1).to(self.device)
        # calculate the gradient of u_theta
        Du = self.Nu1(x_bd).reshape(n, -1)
        # calculate the integral by summing up the dot product of Du and bd_dir
        integrate_bd = torch.sum(Du*bd_dir, 1)/(2*self.Eq.bdsize)
        return integrate_bd
    
    def integrate_BD2(self, X, x_bd, bd_dir):
        n = len(X)
        integrate_bd = torch.zeros(n, 1).to(self.device)
        # calculate the gradient of u_theta
        Du = self.Nu2(x_bd).reshape(n, -1)
        # calculate the integral by summing up the dot product of Du and bd_dir
        integrate_bd = torch.sum(Du*bd_dir, 1)/(2*self.Eq.bdsize)
        return integrate_bd

    # calculate the integral of f in the neighborhood of x
    def integrate_F1(self, X):
        n = len(X)
        integrate_f = torch.zeros([n, 1]).to(self.device)
        for i in range(n):
            x_neighbor     = self.Eq.neighborhood(X[i], 1)
            res            = self.Eq.f1(x_neighbor,self.model)
            integrate_f[i] = torch.mean(res)*self.Eq.E
        return integrate_f.detach().requires_grad_(True)
    
    def integrate_F2(self, X):
        n = len(X)
        integrate_f = torch.zeros([n, 1]).to(self.device)
        for i in range(n):
            x_neighbor     = self.Eq.neighborhood(X[i], 1)
            res            = self.Eq.f2(x_neighbor)
            integrate_f[i] = torch.mean(res)*self.Eq.E
        return integrate_f.detach().requires_grad_(True)

    # Boundary loss function
    def loss_boundary1(self, x_boundary):
        u_theta    = self.model(x_boundary)[:,0].reshape(-1,1)
        u_bound    = self.Eq.g1(x_boundary).reshape(-1,1)
        loss_bd    = LOSS_FN(u_theta, u_bound) 
        return loss_bd
    
    def loss_boundary2(self, x_boundary):
        u_theta    = self.model(x_boundary)[:,1].reshape(-1,1)
        u_bound    = self.Eq.g2(x_boundary).reshape(-1,1)
        loss_bd    = LOSS_FN(u_theta, u_bound) 
        return loss_bd

    # Test function
    def TEST(self, NUM_TESTING):
        with torch.no_grad():
            x_test = torch.Tensor(NUM_TESTING, self.Eq.D).uniform_(a[0], b[0]).requires_grad_(True).to(self.device)
            u_real = self.Eq.u_exact(x_test).reshape(1,-1)
            u_pred = self.model(x_test).reshape(1,-1)
            Error  =  u_real - u_pred
            L2error  = torch.sqrt( torch.mean(Error*Error) )/ torch.sqrt( torch.mean(u_real*u_real) )
            MaxError = torch.max(torch.abs(Error))
        return L2error.cpu().detach().numpy(), MaxError.cpu().detach().numpy()



class MLP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_width=40):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x.to(torch.float32))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_pipeline():
    # define device
    DEVICE = torch.device(f"cuda:{CUDA_ORDER}" if torch.cuda.is_available() else "cpu")
    print(f"Using {DEVICE}")
    # define equation
    Eq = PoissonEQuation(DIMENSION, EPSILON, BDSIZE, DEVICE)
    model = MLP(DIM_INPUT, DIM_OUTPUT, NUM_UNIT).to(DEVICE)
    # file_path = "D:\\workspace\\code_workspace\\DFVM\\PossionEQ_seed0\\2DIM-DFVM_net"
    # model.load_state_dict(torch.load(file_path,map_location=DEVICE))
    optA   = torch.optim.Adam(model.parameters(), lr=LEARN_RATE) 
    solver = DFVMsolver(Eq, model, DEVICE)

    x      = Eq.interior(NUM_TRAIN_SAMPLE)
    x_bd   = Eq.neighborhoodBD(x)
    print(x_bd.shape)
    int_f1  = solver.integrate_F1(x)
    int_f2  = solver.integrate_F2(x)
    bd_dir = Eq.outerNormalVec()
    x_boundary = Eq.boundary(NUM_BOUND_SAMPLE)
    
    # Networks Training
    elapsed_time     = 0  
    training_history = [] 

    for step in tqdm(range(NUM_ITERATION+1)):
        if IS_DECAY and step and step % LEARN_FREQUENCY == 0:
            for p in optA.param_groups:
                if p['lr'] > LEARN_LOWWER_BOUND:
                    p['lr'] = p['lr']*LEARN_DECAY_RATE
                    # print(f"Learning Rate: {p['lr']}")

        start_time = time.time()
        int_bd1   = solver.integrate_BD1(x, x_bd, bd_dir).reshape(-1,1)
        int_bd2   = solver.integrate_BD2(x, x_bd, bd_dir).reshape(-1,1)
        loss_int1 = LOSS_FN(-int_bd1, int_f1)
        loss_int2 = LOSS_FN(-int_bd2, int_f2)
        loss_bd1  = solver.loss_boundary1(x_boundary)
        loss_bd2  = solver.loss_boundary2(x_boundary)
        loss     = loss_int1+ loss_int2 + BETA*loss_bd1 + BETA*loss_bd2
        optA.zero_grad()
        loss.backward()
        optA.step()
        
        epoch_time = time.time() - start_time
        elapsed_time = elapsed_time + epoch_time
        if step % TEST_FREQUENCY == 0:
            loss_int1     = loss_int1.cpu().detach().numpy()
            loss_bd1      = loss_bd1.cpu().detach().numpy()
            loss_int2     = loss_int2.cpu().detach().numpy()
            loss_bd2      = loss_bd2.cpu().detach().numpy()
            loss         = loss.cpu().detach().numpy()
        #     L2error,ME   = solver.TEST(NUM_TEST_SAMPLE)
            if step and step%1000 == 0:
                # tqdm.write( f'\nStep: {step:>5}, '
                #             f'Loss_int: {loss_int:>10.5f}, '
                #             f'Loss_bd: {loss_bd:>10.5f}, '
                #             f'Loss: {loss:>10.5f}, '                                     
                #             f'L2 error: {L2error:.5f}, '                                     
                #             f'Time: {elapsed_time:.2f}')
                tqdm.write( f'\nStep: {step:>5}, '
                            f'Loss_int1: {loss_int1:>10.5f}, '
                            f'Loss_bd1: {loss_bd1:>10.5f}, '
                            f'Loss_int2: {loss_int2:>10.5f}, '
                            f'Loss_bd2: {loss_bd2:>10.5f}, '
                            f'Loss: {loss:>10.5f}, '                                                                          
                            f'Time: {elapsed_time:.2f}')
            training_history.append([step, loss, elapsed_time])

    training_history = np.array(training_history)
    print(np.min(training_history[:,1]))
    print(np.min(training_history[:,2]))

    save_time = time.localtime()
    save_time = f'[{save_time.tm_mday:0>2d}{save_time.tm_hour:0>2d}{save_time.tm_min:0>2d}]'
    dir_path  = os.getcwd() + f'/PossionEQ_seed{seed}/'
    file_name = f'{DIMENSION}DIM-DFVM-{BETA}weight-{NUM_ITERATION}itr-{EPSILON}R-{BDSIZE}bd-{LEARN_RATE}lr.csv'
    file_path = dir_path + file_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.savetxt(file_path, training_history,
                delimiter =",",
                header    ="step,  loss, elapsed_time",
                comments  ='')
    print('Training History Saved!')

    if IS_SAVE_MODEL:
        torch.save(model.state_dict(), dir_path + f'{DIMENSION}DIM-DFVM_net')
        print('DFVM Network Saved!')


if __name__ == "__main__":
    setup_seed(seed)
    train_pipeline()
