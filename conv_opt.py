import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import os
from teleportation import teleport_rotation


def obj_func(x: float | Tensor, y: float | Tensor) -> float | Tensor:
    return x ** 2 + 9 * y ** 2

def xy_to_uv(x: float | Tensor, y: float | Tensor) -> float | Tensor:
    return x, 3 * y

def uv_to_xy(u: float | Tensor, v: float | Tensor) -> float | Tensor:
    return u, v / 3

def plot_level_sets() -> None:
    L: np.ndarray = np.array([1e0, np.sqrt(10) * 1e0, 1e1, np.sqrt(10) * 1e1,
                              1e2, np.sqrt(10) * 1e2, 1e3, np.sqrt(10) * 1e3,
                              1e4])
    t: np.ndarray = np.linspace(0, 2 * np.pi, 1000)
    for loss in L:
        u: float = np.sqrt(loss) * np.cos(t)
        v: float = np.sqrt(loss) * np.sin(t)
        x, y = uv_to_xy(u, v)
        plt.plot(x, y, color='gray')
    plt.xlim(-10.0, 10.0)
    plt.ylim(-10.0, 10.0)

def train_epoch_SGD(x: Tensor, y: Tensor, lr: float) \
    -> tuple[Tensor, Tensor, Tensor, Tensor]:
    L: Tensor = obj_func(x, y)
    dL_dx, dL_dy = torch.autograd.grad(L, inputs=[x, y])
    x_updated: Tensor = x - lr * dL_dx
    y_updated: Tensor = y - lr * dL_dy
    dL_dt: Tensor = dL_dx ** 2 + dL_dy ** 2
    return x_updated, y_updated, L, dL_dt


x0, y0 = 8., -2.  # initialization of parameters
lr: float = 0.08

# gradient descent with no teleportation
x: Tensor = torch.tensor(x0, requires_grad=True)
y: Tensor = torch.tensor(y0, requires_grad=True)
x_arr: list[ndarray] = [x0]  # save trajectory for plotting
y_arr: list[ndarray] = [y0]
loss_arr_sgd: list[ndarray] = []
dL_dt_arr_sgd: list[ndarray] = []
for epoch in range(10):
    x, y, loss, dL_dt = train_epoch_SGD(x, y, lr)
    x_arr.append(x.detach().numpy())
    y_arr.append(y.detach().numpy())
    loss_arr_sgd.append(loss.detach().numpy())
    dL_dt_arr_sgd.append(dL_dt.detach().numpy())

# gradient descent with teleportation
lr_theta: float = 0.001  # for gradient ascent on theta
x: Tensor = torch.tensor(x0, requires_grad=True)
y: Tensor = torch.tensor(y0, requires_grad=True)
x_arr_teleport: list[ndarray] = [x0]
y_arr_teleport: list[ndarray] = [y0]
loss_arr_teleport: list[ndarray] = []
dL_dt_arr_teleport: list[ndarray] = []

for epoch in range(10):
    if epoch == 4:
        x, y = teleport_rotation(x, y, xy_to_uv, uv_to_xy, obj_func, lr_theta)

    x, y, loss, dL_dt = train_epoch_SGD(x, y, lr)
    x_arr_teleport.append(x.detach().numpy())
    y_arr_teleport.append(y.detach().numpy())
    loss_arr_teleport.append(loss.detach().numpy())
    dL_dt_arr_teleport.append(dL_dt.detach().numpy())


# make figures
if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/conv_opt'):
    os.mkdir('figures/conv_opt')

# visualization of GD without teleportation
plt.figure()
plot_level_sets()
plt.scatter(x_arr, y_arr)
plt.plot(x_arr, y_arr)
plt.scatter(x_arr[-1], y_arr[-1], marker="*", s=60, color='#1f77b4')
plt.scatter(0, 0, marker="*", s=40, color='#2ca02c')
plt.savefig('figures/conv_opt/conv_opt_level_set_GD.pdf', bbox_inches='tight')

# visualization of GD with teleportation
x_arr_teleport = np.array(x_arr_teleport)
y_arr_teleport = np.array(y_arr_teleport)
loss_arr_teleport = np.array(loss_arr_teleport)
dL_dt_arr_teleport = np.array(dL_dt_arr_teleport)
loss_arr_sgd = np.array(loss_arr_sgd)
dL_dt_arr_sgd = np.array(dL_dt_arr_sgd)

plt.figure()
plot_level_sets()
plt.scatter(x_arr_teleport, y_arr_teleport, s=20)
plt.plot(x_arr_teleport, y_arr_teleport)
g_idx: ndarray = np.arange(1) * 5 + 4
plt.scatter(x_arr_teleport[g_idx], y_arr_teleport[g_idx], s=20)  # orange dots
for idx in g_idx:
    plt.plot(x_arr_teleport[idx:idx+2], y_arr_teleport[idx:idx+2],
             color='#ff7f0e')  # orange line
plt.scatter(x_arr_teleport[-1], y_arr_teleport[-1], marker="*", s=60,
            color='#1f77b4')  # initial point (blue dot)
plt.scatter(0, 0, marker="*", s=40,
            color='#2ca02c')  # target point (green star)
plt.savefig('figures/conv_opt/conv_opt_level_set_teleport.pdf',
            bbox_inches='tight')

# plot loss vs epoch
plt.figure()
plt.plot(loss_arr_sgd, label='GD', zorder=3)
plt.plot(loss_arr_teleport, label='GD + teleport', zorder=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('figures/conv_opt/conv_opt_loss.pdf', bbox_inches='tight')

# plot dL/dt vs epoch
plt.figure()
plt.plot(dL_dt_arr_sgd, label='GD', zorder=3)
plt.plot(dL_dt_arr_teleport, label='GD + teleport', zorder=2)
plt.xlabel('Epoch')
plt.ylabel('dL/dt')
plt.yscale('log')
plt.legend()
plt.savefig('figures/conv_opt/conv_opt_loss_gradient.pdf', bbox_inches='tight')

# plot loss vs dL/dt
plt.figure()
plt.scatter(loss_arr_sgd[g_idx], dL_dt_arr_sgd[g_idx], s=60)
plt.scatter(loss_arr_teleport[g_idx], dL_dt_arr_teleport[g_idx], s=60,
            label='teleportation point')
plt.plot(loss_arr_sgd[0:], dL_dt_arr_sgd[0:], label='GD', zorder=3)
plt.plot(loss_arr_teleport[0:], dL_dt_arr_teleport[0:],
         label='GD + teleport', zorder=2)
plt.xlabel('Loss')
plt.ylabel('dL/dt')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig('figures/conv_opt/conv_opt_loss_vs_gradient.pdf',
            bbox_inches='tight')
