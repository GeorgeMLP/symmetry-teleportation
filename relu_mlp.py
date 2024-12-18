import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
import time
import torch
from torch import nn, Tensor
import os
from typing import Callable
from teleportation import teleport_relu_mlp


sigma: Callable[[Tensor], Tensor] = nn.ReLU()

# list of dimensions of weight matrices
# example: [4, 5, 6, 7, 8] -> X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
dim: list[int] = [4, 5, 6, 7, 8] 

def init_param(dim: list[int], seed: int = 12345) \
    -> tuple[list[Tensor], Tensor, Tensor]:
    torch.manual_seed(seed)
    W_list: list[Tensor] = []
    for i in range(len(dim) - 2):
        W_list.append(torch.rand(dim[i+2], dim[i+1], requires_grad=True))
    X: Tensor = torch.rand(dim[1], dim[0], requires_grad=True)
    Y: Tensor = torch.rand(dim[-1], dim[0], requires_grad=True)
    return W_list, X, Y

def loss_multi_layer(W_list: list[Tensor], X: Tensor, Y: Tensor) -> Tensor:
    h: Tensor = X
    for i in range(len(W_list) - 1):
        h = sigma(torch.matmul(W_list[i], h))
    return 0.5 * torch.norm(Y - torch.matmul(W_list[-1], h)) ** 2

def train_epoch_gd(W_list: list[Tensor], X: Tensor, Y: Tensor, lr: float) \
    -> tuple[list[Tensor], Tensor, Tensor]:
    L: Tensor = loss_multi_layer(W_list, X, Y)
    dL_dW_list: list[Tensor] = torch.autograd.grad(L, inputs=W_list)
    W_list_updated: list[Tensor] = []
    dL_dt: Tensor = 0
    for i in range(len(W_list)):
        W_list_updated.append(W_list[i] - lr * dL_dW_list[i])
        dL_dt += torch.norm(dL_dW_list[i])**2 
    return W_list_updated, L, dL_dt

def train_epoch_adagrad(W_list: list[Tensor], X: Tensor, Y: Tensor, lr: float,
                        G_list: list[Tensor], eps: float = 1e-10) -> \
                            tuple[list[Tensor], Tensor, Tensor, list[Tensor]]:
    L: Tensor = loss_multi_layer(W_list, X, Y)
    dL_dW_list: list[Tensor] = torch.autograd.grad(L, inputs=W_list)
    W_list_updated: list[Tensor] = []
    dL_dt: Tensor = 0
    for i in range(len(W_list)):
        G_list[i] = G_list[i] + dL_dW_list[i] * dL_dW_list[i]
        W_list_updated.append(W_list[i] - lr * torch.div(
            dL_dW_list[i], torch.sqrt(G_list[i]) + eps)
        )
        dL_dt += torch.norm(dL_dW_list[i])**2 
    return W_list_updated, L, dL_dt, G_list


# do some random things first so that wall-clock time comparison is fair
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n * n * 12345)
    lr: float = 1e-4
    for epoch in range(300):
        W_list, loss, dL_dt = train_epoch_gd(W_list, X, Y, lr)

# training with GD
print("GD")
time_arr_sgd_n: list[list[float]] = []
loss_arr_sgd_n: list[list[ndarray]] = []
dL_dt_arr_sgd_n: list[list[ndarray]] = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n * n * 12345)
    lr: float = 1e-4
    time_arr_sgd: list[float] = []
    loss_arr_sgd: list[ndarray] = []
    dL_dt_arr_sgd: list[ndarray] = []

    t0: float = time.time()
    for epoch in range(300):
        W_list, loss, dL_dt = train_epoch_gd(W_list, X, Y, lr)
        t1: float = time.time()
        time_arr_sgd.append(t1 - t0)
        loss_arr_sgd.append(loss.detach().numpy())
        dL_dt_arr_sgd.append(dL_dt.detach().numpy())

    time_arr_sgd_n.append(time_arr_sgd)
    loss_arr_sgd_n.append(loss_arr_sgd)
    dL_dt_arr_sgd_n.append(dL_dt_arr_sgd)


# training with teleportation
print("GD and teleportation")
time_arr_sgd_teleport_n: list[list[float]] = []
loss_arr_sgd_teleport_n: list[list[ndarray]] = []
dL_dt_arr_sgd_teleport_n: list[list[ndarray]] = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n * n * 12345)
    lr: float = 1e-4
    lr_teleport: float = 1e-4
    time_arr_sgd_teleport: list[float] = []
    loss_arr_sgd_teleport: list[ndarray] = []
    dL_dt_arr_sgd_teleport: list[ndarray] = []

    t0: float = time.time()
    for epoch in range(300):
        if epoch == 5:
            W_list = teleport_relu_mlp(W_list, X, Y, lr_teleport, dim,
                                       loss_multi_layer, 10, 0.1)

        W_list, loss, dL_dt = train_epoch_gd(W_list, X, Y, lr)
        t1: float = time.time()
        time_arr_sgd_teleport.append(t1 - t0)
        loss_arr_sgd_teleport.append(loss.detach().numpy())
        dL_dt_arr_sgd_teleport.append(dL_dt.detach().numpy())

    time_arr_sgd_teleport_n.append(time_arr_sgd_teleport)
    loss_arr_sgd_teleport_n.append(loss_arr_sgd_teleport)
    dL_dt_arr_sgd_teleport_n.append(dL_dt_arr_sgd_teleport)


# training with AdaGrad
print("AdaGrad")
time_arr_adagrad_n: list[list[float]] = []
loss_arr_adagrad_n: list[list[ndarray]] = []
dL_dt_arr_adagrad_n: list[list[ndarray]] = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n * n * 12345)
    lr: float = 1e-1
    time_arr_adagrad: list[float] = []
    loss_arr_adagrad: list[ndarray] = []
    dL_dt_arr_adagrad: list[ndarray] = []

    G_list: list[Tensor] = []
    for i in range(len(W_list)):
        G_list.append(torch.zeros_like(W_list[i]))

    t0: float = time.time()
    for epoch in range(300):
        W_list, loss, dL_dt, G_list = train_epoch_adagrad(W_list, X, Y,
                                                          lr, G_list)
        t1: float = time.time()
        time_arr_adagrad.append(t1 - t0)
        loss_arr_adagrad.append(loss.detach().numpy())
        dL_dt_arr_adagrad.append(dL_dt.detach().numpy())

    time_arr_adagrad_n.append(time_arr_adagrad)
    loss_arr_adagrad_n.append(loss_arr_adagrad)
    dL_dt_arr_adagrad_n.append(dL_dt_arr_adagrad)


# training with AdaGrad and teleportation
print("AdaGrad and teleportation")
time_arr_adagrad_teleport_n: list[list[float]] = []
loss_arr_adagrad_teleport_n: list[list[ndarray]] = []
dL_dt_arr_adagrad_teleport_n: list[list[ndarray]] = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n * n * 12345)
    lr: float = 1e-1
    lr_teleport: float = 1e-5
    time_arr_adagrad_teleport: list[float] = []
    loss_arr_adagrad_teleport: list[ndarray] = []
    dL_dt_arr_adagrad_teleport: list[ndarray] = []

    G_list: list[Tensor] = []
    for i in range(len(W_list)):
        G_list.append(torch.zeros_like(W_list[i]))

    t0: float = time.time()
    for epoch in range(300):
        if epoch == 5:
            W_list = teleport_relu_mlp(W_list, X, Y, lr_teleport, dim,
                                       loss_multi_layer, 10, 0.1)

        W_list, loss, dL_dt, G_list = train_epoch_adagrad(W_list, X, Y,
                                                          lr, G_list)
        t1: float = time.time()
        time_arr_adagrad_teleport.append(t1 - t0)
        loss_arr_adagrad_teleport.append(loss.detach().numpy())
        dL_dt_arr_adagrad_teleport.append(dL_dt.detach().numpy())

    time_arr_adagrad_teleport_n.append(time_arr_adagrad_teleport)
    loss_arr_adagrad_teleport_n.append(loss_arr_adagrad_teleport)
    dL_dt_arr_adagrad_teleport_n.append(dL_dt_arr_adagrad_teleport)

loss_arr_sgd_n: ndarray = np.array(loss_arr_sgd_n)
dL_dt_arr_sgd_n: ndarray = np.array(dL_dt_arr_sgd_n)
loss_arr_sgd_teleport_n: ndarray = np.array(loss_arr_sgd_teleport_n)
dL_dt_arr_sgd_teleport_n: ndarray = np.array(dL_dt_arr_sgd_teleport_n)
loss_arr_adagrad_n: ndarray = np.array(loss_arr_adagrad_n)
dL_dt_arr_adagrad_n: ndarray = np.array(dL_dt_arr_adagrad_n)
loss_arr_adagrad_teleport_n: ndarray = np.array(loss_arr_adagrad_teleport_n)
dL_dt_arr_adagrad_teleport_n: ndarray = np.array(dL_dt_arr_adagrad_teleport_n)

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/relu'):
    os.mkdir('figures/relu')

# plot loss vs epoch
plt.figure()
mean_sgd: ndarray = np.mean(loss_arr_sgd_n, axis=0)
std_sgd: ndarray = np.std(loss_arr_sgd_n, axis=0)
mean_sgd_teleport: ndarray = np.mean(loss_arr_sgd_teleport_n, axis=0)
std_sgd_teleport: ndarray = np.std(loss_arr_sgd_teleport_n, axis=0)
mean_adagrad: ndarray = np.mean(loss_arr_adagrad_n, axis=0)
std_adagrad: ndarray = np.std(loss_arr_adagrad_n, axis=0)
mean_adagrad_teleport: ndarray = np.mean(loss_arr_adagrad_teleport_n, axis=0)
std_adagrad_teleport: ndarray = np.std(loss_arr_adagrad_teleport_n, axis=0)
plt.plot(mean_sgd, label='GD')
plt.plot(mean_sgd_teleport, label='GD + teleport')
plt.plot(mean_adagrad, label='AdaGrad')
plt.plot(mean_adagrad_teleport, label='AdaGrad + teleport')
plt.gca().set_prop_cycle(None)
plt.fill_between(np.arange(300), mean_sgd - std_sgd,
                 mean_sgd + std_sgd, alpha=0.5)
plt.fill_between(np.arange(300), mean_sgd_teleport - std_sgd_teleport,
                 mean_sgd_teleport + std_sgd_teleport, alpha=0.5)
plt.fill_between(np.arange(300), mean_adagrad - std_adagrad,
                 mean_adagrad + std_adagrad, alpha=0.5)
plt.fill_between(np.arange(300), mean_adagrad_teleport - std_adagrad_teleport,
                 mean_adagrad_teleport + std_adagrad_teleport, alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.xticks([0, 100, 200, 300])
plt.legend()
plt.savefig('figures/relu/multi_layer_loss.pdf', bbox_inches='tight')

# plot loss vs wall-clock time
plt.figure()
g_idx: ndarray = np.arange(10) * 10 + 9
g_idx.astype(int)
plt.plot(np.mean(time_arr_sgd_n, axis=0), mean_sgd, label='GD')
plt.plot(np.mean(time_arr_sgd_teleport_n, axis=0), mean_sgd_teleport,
         label='GD + teleport')
plt.plot(np.mean(time_arr_adagrad_n, axis=0), mean_adagrad, label='AdaGrad')
plt.plot(np.mean(time_arr_adagrad_teleport_n, axis=0), mean_adagrad_teleport,
         label='AdaGrad + teleport')
plt.fill_between(np.mean(time_arr_sgd_n, axis=0), mean_sgd - std_sgd,
                 mean_sgd + std_sgd, alpha=0.5)
plt.fill_between(np.mean(time_arr_sgd_teleport_n, axis=0),
                 mean_sgd_teleport - std_sgd_teleport,
                 mean_sgd_teleport + std_sgd_teleport, alpha=0.5)
plt.fill_between(np.mean(time_arr_adagrad_n, axis=0),
                 mean_adagrad - std_adagrad,
                 mean_adagrad + std_adagrad, alpha=0.5)
plt.fill_between(np.mean(time_arr_adagrad_teleport_n, axis=0),
                 mean_adagrad_teleport - std_adagrad_teleport,
                 mean_adagrad_teleport + std_adagrad_teleport, alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Loss')
max_t: float = np.max(time_arr_sgd_teleport_n)
interval: float = np.round(max_t * 0.3, 2)
plt.xticks([0, interval, interval * 2, interval * 3])
plt.yscale('log')
plt.legend()
plt.savefig('figures/relu/multi_layer_loss_vs_time.pdf', bbox_inches='tight')

# plot dL/dt vs epoch
plt.figure()
plt.plot(dL_dt_arr_sgd, label='GD')
plt.plot(dL_dt_arr_sgd_teleport, label='GD + teleport')
plt.plot(dL_dt_arr_adagrad, label='AdaGrad')
plt.plot(dL_dt_arr_adagrad_teleport, label='AdaGrad + teleport')
plt.xlabel('Epoch')
plt.ylabel('dL/dt')
plt.yscale('log')
plt.xticks([0, 100, 200, 300])
plt.yticks([1e1, 1e3, 1e5, 1e7])
plt.legend()
plt.savefig('figures/relu/multi_layer_loss_gradient.pdf', bbox_inches='tight')

# plot loss vs dL/dt
plt.figure()
n: int = 0
g_idx: ndarray = np.arange(10) * 10 + 9
g_idx.astype(int)
plt.plot(loss_arr_sgd_n[n], dL_dt_arr_sgd_n[n], label='GD')
plt.plot(loss_arr_sgd_teleport_n[n], dL_dt_arr_sgd_teleport_n[n],
         label='GD + teleport')
plt.plot(loss_arr_adagrad_n[n], dL_dt_arr_adagrad_n[n], label='AdaGrad')
plt.plot(loss_arr_adagrad_teleport_n[n], dL_dt_arr_adagrad_teleport_n[n],
         label='AdaGrad + teleport')
plt.xlabel('Loss')
plt.ylabel('dL/dt')
plt.yscale('log')
plt.xscale('log')
plt.yticks([1e1, 1e3, 1e5, 1e7])
plt.legend()
plt.savefig('figures/relu/multi_layer_loss_vs_gradient.pdf',
            bbox_inches='tight')
