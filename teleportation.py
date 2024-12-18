import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import relu
from typing import Callable


def apply_rotation(x: float | Tensor, y: float | Tensor, g: Tensor,
                   xy_to_uv: Callable, uv_to_xy: Callable) -> \
                    tuple[Tensor, Tensor]:
    """
    Apply rotation on a 2D vector.

    Parameters
    ----------
    x : float | Tensor
        The `x` coordinate of the vector.
    y : float | Tensor
        The `y` coordinate of the vector.
    g : Tensor
        A 2D rotation matrix.
    xy_to_uv : Callable
        A function that transforms `x,y` to `u,v`, where `g` acts on `u,v`.
    uv_to_xy : Callable
        A function that transforms `u,v` back to `x,y`.
    
    Returns
    -------
    tuple[Tensor, Tensor]
        The `x` and `y` coordinates of the transformed 2D vector.
    """
    u, v = xy_to_uv(x, y)
    u_v: Tensor = torch.einsum('ij,j->i', g, torch.stack((u, v)))
    u: Tensor = u_v[0]
    v: Tensor = u_v[1]
    return uv_to_xy(u, v)


def teleport_rotation(x: float | Tensor, y: float | Tensor,
                      xy_to_uv: Callable, uv_to_xy: Callable,
                      loss_func: Callable, lr_theta: float,
                      teleport_step: int = 10):
    """
    Teleportation on a function with rotational symmetry.

    Parameters
    ----------
    x : float | Tensor
        The `x` coordinate of the input vector.
    y : float | Tensor
        The `y` coordinate of the input vector.
    xy_to_uv : Callable
        A function that transforms `x,y` to `u,v`, where `g` acts on `u,v`.
    uv_to_xy : Callable
        A function that transforms `u,v` back to `x,y`.
    loss_func : Callable
        The objective function that has rotational symmetry on `u,v`.
    lr_theta : float
        Learning rate of the rotational angle `theta`.
    teleport_step : int
        Number of steps of the symmetry teleportation.
    
    Returns
    -------
    tuple[Tensor, Tensor]
        The `x` and `y` coordinates of the teleported vector.
    """
    theta: Tensor = torch.tensor(np.random.random() * np.pi, requires_grad=True)
    for _ in range(teleport_step):
        g: Tensor = torch.vstack([
            torch.cat([torch.cos(theta).view(1), -torch.sin(theta).view(1)]),
            torch.cat([torch.sin(theta).view(1), torch.cos(theta).view(1)])
        ])
        gx, gy = apply_rotation(x, y, g, xy_to_uv, uv_to_xy)

        L: Tensor = loss_func(gx, gy)
        dL_dgW: list[Tensor] = torch.autograd.grad(L, inputs=[gx, gy],
                                                   create_graph=True)
        dL_dt: Tensor = torch.square(dL_dgW[0]) + torch.square(dL_dgW[1])
        dLdt_dtheta: Tensor = torch.autograd.grad(dL_dt, inputs=[theta])[0]

        theta = theta + lr_theta * dLdt_dtheta

    x: Tensor = torch.tensor(gx.detach().numpy(), requires_grad=True)
    y: Tensor = torch.tensor(gy.detach().numpy(), requires_grad=True)
    return x, y


def apply_positive_scaling(d: Tensor, W1: Tensor, W2: Tensor) \
    -> tuple[Tensor, Tensor]:
    """
    Apply the positive scale group action on weight matrices `W1` and `W2` from
    adjacent layers.

    Parameters
    ----------
    d : Tensor
        The vector of diagonal entries of the transformation matrix `D`.
    W1 : Tensor
        Weight matrix of the `l-1`th layer, with shape `d_l * d_{l-1}`.
    W2 : Tensor
        Weight matrix of the `l`th layer, with shape `d_{l+1} * d_l`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Tuple of the transformed weight matrices `D^{-1} W_1` and `W_2 D`.
    
    Examples
    --------
    >>> W1 = torch.randn(4, 3)
    >>> W2 = torch.randn(2, 4)
    >>> x = torch.randn(3)
    >>> y1 = W2 @ relu(W1 @ x)
    >>> d = torch.Tensor([1, 2, 3, 4])
    >>> W1, W2 = apply_positive_scaling(d, W1, W2)
    >>> y2 = W2 @ relu(W1 @ x)
    >>> torch.allclose(y1, y2)
    True
    """
    assert d.shape[0] == W1.shape[0], 'Shapes of d and W1 do not match'
    assert d.shape[0] == W2.shape[1], 'Shapes of d and W2 do not match'
    D: Tensor = torch.diag(d)
    D_inv: Tensor = torch.diag(1 / d)
    W1 = D_inv @ W1
    W2 = W2 @ D
    return W1, W2


def teleport_relu_mlp(W_list: list[Tensor], x: Tensor, y: Tensor,
                      lr_teleport: float, dim: list[int], loss_func: Callable,
                      teleport_step: int = 10, clamp_ratio: float = 0.99) \
                        -> list[Tensor]:
    """
    Teleportation on a ReLU neural network with positive-scale invariance.

    Parameters
    ----------
    W_list : list[Tensor]
        Weight matrices of the MLP network.
    x : Tensor
        Inputs tensor of the MLP network.
    y : Tensor
        Labels of the input `x`.
    lr_teleport : float
        Learning rate of the symmetry teleportation.
    dim : list[int]
        A list of integers representing the input, hidden and output dimensions
        of the MLP network.
    loss_func : Callable[[list[Tensor], Tensor, Tensor], Tensor]
        A function for forward propagating the network and computing the loss.
    teleport_step : int
        Number of steps of the symmetry teleportation.
    clamp_ratio : float
        Ratio of `1/lr_teleport` to clamp the gradients of group actions.
    
    Returns
    -------
    list[Tensor]
        The teleported weight matrices of the MLP network.
    """
    params: list[Tensor] = W_list.copy()
    device: torch.device = x.device
    num_pairs: int = len(dim) - 3
    action_shapes: list[int] = [dim[i] for i in range(2, 2 + num_pairs)]
    weight_pair_indices: list[list[int]] = [[i, i + 1] for i in range(num_pairs)]
    for _ in range(teleport_step):
        d_lst: list[Tensor] = [torch.ones(action_shapes[i], requires_grad=True, device=device)
                               for i in range(num_pairs)]
        new_params: list[Tensor] = params.copy()
        for d, pair in zip(d_lst, weight_pair_indices):
            new_params[pair[0]], new_params[pair[1]] = apply_positive_scaling(
                d, new_params[pair[0]], new_params[pair[1]]
            )
        loss: Tensor = loss_func(new_params, x, y)
        dL_dW: list[Tensor] = torch.autograd.grad(loss, inputs=new_params,
                                                  create_graph=True)
        dL_dt = 0
        for g in dL_dW:
            dL_dt += torch.norm(g) ** 2
        dg_dd = torch.autograd.grad(dL_dt, inputs=d_lst)
        dg_dd = [torch.clamp(dg_dd_i, -clamp_ratio / lr_teleport, clamp_ratio / lr_teleport)
                 for dg_dd_i in dg_dd]
        for i in range(len(d_lst)):
            d_lst[i] = d_lst[i] + lr_teleport * dg_dd[i]
        for d, pair in zip(d_lst, weight_pair_indices):
            params[pair[0]], params[pair[1]] = apply_positive_scaling(
                d, params[pair[0]], params[pair[1]]
            )
    return [p.detach().requires_grad_() for p in params]
