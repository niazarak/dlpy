from typing import List, Tuple

import numpy as np

from utils import fix_seed


def L_gen(net, loss_mod, ws, x, y):
    parameters = net.parameters()
    for param, w in zip(parameters, ws):
        param[:] = w[1]
    res = net.forward(*[p for name, p in x])

    if type(res) == tuple:
        res = res[0]

    loss = loss_mod.forward(res, y)
    return loss


def L_grad_gen(net, loss_mod, ws, x, y):
    for g in net.grads():
        g[:] = np.zeros_like(g)
    assert all((g == 0).all() for g in net.grads())
    parameters = net.parameters()
    for param, w in zip(parameters, ws):
        param[:] = w[1]
    res = net.forward(*[p for name, p in x])

    if type(res) == tuple:
        res = res[0]

    loss_mod.forward(res, y)

    dx = loss_mod.backward()
    # import pdb; pdb.set_trace()
    dx = net.backward(dx)
    if type(dx) != tuple:
        dx = (dx,)
    return list(dx) + net.grads()


def copy_params(w: List[np.ndarray]):
    return [np.copy(_w) for _w in w]


def copy_named_params(w: List[Tuple[str, np.ndarray]]):
    return [(p, np.copy(_w)) for p, _w in w]


def grad_check(J, grad_J, w, eps=1e-06, tol=1e-4):
    for p_i, param in enumerate(w):
        it = np.nditer(param[1], flags=['multi_index'])
        while not it.finished:
            fix_seed(0)
            param_grad = grad_J(_w=w)[p_i]

            try:
                assert param_grad.shape[-len(param[1].shape):] == param[1].shape
            except:
                import pdb;
                pdb.set_trace()
            grad = param_grad[it.multi_index]

            delta1 = copy_named_params(w)
            delta1[p_i][1][it.multi_index] -= eps
            fix_seed(0)
            f1 = J(_w=delta1)

            delta2 = copy_named_params(w)
            delta2[p_i][1][it.multi_index] += eps
            fix_seed(0)
            f2 = J(_w=delta2)
            num_grad = (f2 - f1) / (eps * 2)

            assert abs(num_grad - grad) < eps, {'num': num_grad, 'analytical': grad, 'param': param[0]}

            rel_tol = np.abs(num_grad - grad) / (1. + np.minimum(np.abs(num_grad), np.abs(grad)))
            assert rel_tol < tol

            it.iternext()
    print("Gradcheck passed")
