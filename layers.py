from typing import List, Tuple

import numpy as np

SHAPE_BATCH = "B"
SHAPE_SEQ_LEN = "S"
SHAPE_ANY = "A"


class Layer:
    def __init__(self):
        self.is_train = True

    def _sublayers(self):
        layers = []
        for attr in dir(self):
            attr_obj = getattr(self, attr)
            if isinstance(attr_obj, Layer):
                layers.append(attr_obj)

        return layers

    def train(self, value: bool = True):
        self.is_train = value
        for l in self._sublayers():
            l.train(value=value)

    def eval(self):
        self.train(False)


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.weight = np.random.normal(size=(in_features, out_features))
        self.bias = np.random.normal(size=(1, out_features))

        self.last_x = None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        res = x.dot(self.weight) + self.bias
        self.last_x = x
        return res

    def backward(self, dx: np.ndarray):
        assert self.last_x is not None, 'Cannot run backward pass before forward pass.'

        np.dot(self.last_x.T, dx, self.grad_weight)
        np.sum(dx, axis=0, keepdims=True, out=self.grad_bias)
        return dx.dot(self.weight.T)

    def parameters(self):
        return [self.weight, self.bias]

    def named_parameters(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def grads(self):
        return [self.grad_weight, self.grad_bias]

    def __repr__(self) -> str:
        return f'Linear{self.weight.shape}'

    def input_shape(self):
        return [SHAPE_BATCH, self.in_features],


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

    def _zero(self):
        self.grad[:] = np.zeros_like(self.grad)


class LSTMGate(Layer):
    def __init__(self, hidden_size: int, input_features: int, activation):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.weight = Parameter(np.random.normal(size=(input_features, hidden_size)))
        self.bias = Parameter(np.random.normal(size=(1, hidden_size)))
        self.u = Parameter(np.random.normal(size=(hidden_size, hidden_size)))

        self.activation = activation

        self.last_x, self.last_h = [], []

    def forward(self, x, h) -> np.ndarray:
        self.last_x.append(np.copy(x))
        self.last_h.append(np.copy(h))  # The h can be modified in-place, so copy to preserve correct history

        res = x.dot(self.weight.data) + h.dot(self.u.data) + self.bias.data

        res = self.activation.forward(res)
        return res

    def backward(self, dx) -> Tuple[np.ndarray, np.ndarray]:
        assert self.last_x and self.last_h
        dx = self.activation.backward(dx)

        last_x = self.last_x.pop()
        last_h = self.last_h.pop()

        np.add(np.dot(last_x.T, dx), self.weight.grad, out=self.weight.grad)
        np.add(np.dot(last_h.T, dx), self.u.grad, out=self.u.grad)
        np.add(np.sum(dx, axis=0, keepdims=True), self.bias.grad, out=self.bias.grad)

        return dx.dot(self.weight.data.T), dx.dot(self.u.data.T)

    def parameters(self):
        return [p.data for p in [self.weight, self.u, self.bias]]

    def named_parameters(self):
        return [(name, p.data) for name, p in [('weight', self.weight), ('u', self.u), ('bias', self.bias)]]

    def grads(self):
        return [p.grad for p in [self.weight, self.u, self.bias]]

    def input_shape(self):
        return [SHAPE_BATCH, self.input_features], [SHAPE_BATCH, self.hidden_size]


def merge_grads(mods):
    return [p for m in mods for p in m.grads()]


def merge_parameters(mods):
    return [p for m in mods for p in m.parameters()]


def merge_named_parameters(named_mods):
    return [(f'{m_name}.{name}', p) for m_name, m in named_mods for name, p in m.named_parameters()]


class LSTMCell(Layer):
    def __init__(self, hidden_size: int, input_features: int):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size

        self.forget_gate = LSTMGate(hidden_size, input_features, SigmoidFunction())
        self.input_gate = LSTMGate(hidden_size, input_features, SigmoidFunction())
        self.output_gate = LSTMGate(hidden_size, input_features, SigmoidFunction())
        self.cell_gate = LSTMGate(hidden_size, input_features, TanhFunction())

        self.out_act_tanh = TanhFunction()
        self.caches = []

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        ft = self.forget_gate.forward(x, h_prev)
        ot = self.output_gate.forward(x, h_prev)
        it = self.input_gate.forward(x, h_prev)
        c_t = self.cell_gate.forward(x, h_prev)

        ct = ft * c_prev + it * c_t
        ct_act = self.out_act_tanh.forward(ct)
        ht = ot * ct_act

        cache = {
            'x': np.copy(x), 'h_prev': np.copy(h_prev), 'c_prev': np.copy(c_prev),
            'ft': ft, 'it': it, 'ot': ot, 'c_t': c_t, 'ct_act': ct_act,
        }

        self.caches.append(cache)
        return ht, ct

    def backward(self, dh: np.ndarray, dc: np.ndarray):
        cache = self.caches.pop()
        # import pdb; pdb.set_trace()

        dx_ot, dh_prev_ot = self.output_gate.backward(cache['ct_act'] * dh)

        dh_dc = self.out_act_tanh.backward(cache['ot'] * dh)
        dc += dh_dc
        dc_prev = dc * cache['ft']

        dft = dc * cache['c_prev']
        dx_ft, dh_prev_ft = self.forget_gate.backward(dft)

        dit = dc * cache['c_t']
        dx_it, dh_prev_it = self.input_gate.backward(dit)

        dc_t = dc * cache['it']
        dx_c_t, dh_prev_c_t = self.cell_gate.backward(dc_t)

        dx = dx_ft + dx_ot + dx_it + dx_c_t
        dh_prev = dh_prev_ft + dh_prev_ot + dh_prev_it + dh_prev_c_t

        return dh_prev, dc_prev, dx

    def parameters(self):
        return merge_parameters([self.forget_gate, self.output_gate, self.input_gate, self.cell_gate])

    def named_parameters(self):
        return merge_named_parameters([
            ('forget_gate', self.forget_gate),
            ('output_gate', self.output_gate),
            ('input_gate', self.input_gate),
            ('cell_gate', self.cell_gate)
        ])

    def grads(self):
        return merge_grads([self.forget_gate, self.output_gate, self.input_gate, self.cell_gate])


class FlattenBatchLayer(Layer):
    def __init__(self):
        super().__init__()
        self.cache = []

    def forward(self, x):
        assert len(x.shape) in {2, 3}
        self.cache.append(x)
        return x.reshape(-1, x.shape[-1])

    def backward(self, dx):
        assert self.cache
        return dx.reshape(self.cache.pop().shape)

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_SEQ_LEN, SHAPE_ANY],


class SliceLayer(Layer):
    def __init__(self, slice_obj):
        super().__init__()
        self.last_x = None
        self.slice_obj = slice_obj

    def forward(self, x):
        self.last_x = x
        res = x[self.slice_obj]
        return res

    def backward(self, dx):
        res = np.zeros_like(self.last_x)
        res[self.slice_obj] = dx
        return res

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []


class SimpleLSTM(Layer):
    def __init__(self, hidden_size: int, input_features: int):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.cell = LSTMCell(hidden_size, input_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 3  # Batch * Len * Input

        h = np.zeros(shape=(x.shape[0], self.hidden_size))
        c = np.zeros(shape=(x.shape[0], self.hidden_size))

        out = []
        for i in range(x.shape[1]):
            h_new, c_new = self.cell.forward(x[:, i, :], h, c)
            out.append(h_new)
            h[:] = h_new
            c[:] = c_new

        out = np.stack(out, axis=1)

        assert out.shape == (x.shape[0], x.shape[1], self.hidden_size)
        # import pdb; pdb.set_trace()
        return out

    def backward(self, dxs: np.ndarray) -> np.ndarray:
        dh, dc = np.zeros_like(dxs[:, 0, :]), np.zeros_like(dxs[:, 0, :])
        dxs_ = []
        # import pdb; pdb.set_trace()
        for i in range(dxs.shape[1] - 1, -1, -1):
            dh += dxs[:, i, :]
            dh, dc, dx = self.cell.backward(dh, dc)

            dxs_.append(dx)

        dxs_ = np.stack(dxs_[::-1], axis=1)
        if (dxs_ == 0).all():
            import pdb;
            pdb.set_trace()
        return dxs_

    def parameters(self):
        return self.cell.parameters()

    def named_parameters(self):
        return merge_named_parameters([
            ('cell', self.cell)
        ])

    def grads(self):
        return self.cell.grads()

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_SEQ_LEN, self.input_features],


class LSTMSampler(Layer):
    def __init__(self, hidden_size: int, encoder_input_features: int, decoder_input_features: int, teacher_forcing=True):
        super().__init__()
        self.encoder_input_features = encoder_input_features
        self.decoder_input_features = decoder_input_features
        self.hidden_size = hidden_size
        self.encoder_cell = LSTMCell(hidden_size, encoder_input_features)
        self.decoder_cell = LSTMCell(hidden_size, decoder_input_features)

        self.proj_dropout = Dropout(0.1)
        self.proj_layer = Linear(hidden_size, decoder_input_features)
        self.proj_softmax = SoftmaxFunction()

        self.teacher_forcing = teacher_forcing
        self.cache = []

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert len(x.shape) == len(y.shape) == 3
        assert y.shape[1] > 0

        self.cache.append({'x': x.copy(), 'y': y.copy()})
        h = np.zeros(shape=(x.shape[0], self.hidden_size))
        c = np.zeros(shape=(x.shape[0], self.hidden_size))

        for i in range(x.shape[1]):
            h_new, c_new = self.encoder_cell.forward(x[:, i, :], h, c)
            h[:] = h_new
            c[:] = c_new

        out = []
        for i in range(y.shape[1]):
            y_curr: np.ndarray
            if (self.teacher_forcing and self.is_train) or i == 0:
                y_curr = y[:, i, :]
            else:
                y_curr = out[-1]

            h_new, c_new = self.decoder_cell.forward(y_curr, h, c)
            h[:] = h_new
            c[:] = c_new

            y_new = self.forward_proj(h_new)

            out.append(y_new)

        out = np.stack(out, axis=1)

        assert out.shape == (y.shape[0], y.shape[1], self.decoder_input_features)
        return out

    def forward_proj(self, h_new):
        y = self.proj_dropout.forward(h_new)
        y = self.proj_layer.forward(y)
        y = self.proj_softmax.forward(y)
        return y

    def backward(self, dxs: np.ndarray):
        dh = np.zeros(shape=(dxs.shape[0], self.hidden_size))
        dc = np.zeros(shape=(dxs.shape[0], self.hidden_size))

        cache = self.cache.pop()

        dys_ = []
        dy = np.zeros_like(dxs[:, 0, :])
        for i in range(cache['y'].shape[1] - 1, -1, -1):
            dy += dxs[:, i, :]
            dh += self.backward_proj(dy)
            dh, dc, dy = self.decoder_cell.backward(dh, dc)

            if self.teacher_forcing or i == 0:
                dys_.append(dy.copy())
                dy = np.zeros_like(dxs[:, 0, :])
            else:
                dys_.append(np.zeros_like(dxs[:, 0, :]))

        dys_ = np.stack(dys_[::-1], axis=1)

        dxs_ = []
        for i in range(cache['x'].shape[1] - 1, -1, -1):
            dh, dc, dx = self.encoder_cell.backward(dh, dc)
            dxs_.append(dx)

        dxs_ = np.stack(dxs_[::-1], axis=1)
        return dxs_, dys_

    def backward_proj(self, dy):
        dy = self.proj_softmax.backward(dy)
        dy = self.proj_layer.backward(dy)
        dh = self.proj_dropout.backward(dy)
        return dh

    def parameters(self):
        return merge_parameters([self.encoder_cell, self.decoder_cell])

    def named_parameters(self):
        return merge_named_parameters([
            ('encoder_cell', self.encoder_cell),
            ('decoder_cell', self.decoder_cell),
        ])

    def grads(self):
        return merge_grads([self.encoder_cell, self.decoder_cell])

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_SEQ_LEN, self.encoder_input_features],


class LayerList(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def _sublayers(self):
        return self.layers

    def forward(self, *x):
        for m in self.layers:
            x = m.forward(*x)
            if type(x) != tuple:
                x = (x,)
        return x

    def backward(self, dx=None):
        for m in self.layers[::-1]:
            dx = m.backward(dx) if dx is not None else m.backward()
        return dx

    def parameters(self) -> List[np.ndarray]:
        return [p for m in self.layers for p in m.parameters()]

    def named_parameters(self) -> List[Tuple[str, np.ndarray]]:
        return [(f'{m_i}.{name}', p) for m_i, m in enumerate(self.layers) for name, p in m.named_parameters()]

    def grads(self):
        return [p for m in self.layers for p in m.grads()]

    def input_shape(self):
        return self.layers[0].input_shape()


eps = 1e-9


class ReLUFunction(Layer):
    def __init__(self):
        super().__init__()
        self.last_call_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_call_cache = np.clip(x, 0, None)
        return self.last_call_cache

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.last_call_cache > 0)

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_ANY],


class SoftmaxFunction(Layer):
    def __init__(self):
        super().__init__()
        self.cache = []

    def forward(self, x):
        res = softmax(x)
        self.cache.append({'res': res})
        return res

    def backward(self, dx):
        s = self.cache.pop()['res']
        n = s.shape[1]
        jac = np.zeros((s.shape[0], n, n))
        for i in range(n):
            for j in range(n):
                jac[:, i, j] = dx[:, j] * s[:, i] * (int(i == j) - s[:, j])

        return jac.sum(axis=2)

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_ANY],


class TanhFunction(Layer):
    def __init__(self):
        super().__init__()
        self.last_call_cache = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        res = np.tanh(x)
        self.last_call_cache.append(np.copy(res))
        return res

    def backward(self, dx: np.ndarray) -> np.ndarray:
        assert self.last_call_cache
        return (1 - self.last_call_cache.pop() ** 2) * dx

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_ANY],


class SigmoidFunction(Layer):
    def __init__(self):
        super().__init__()
        self.last_call_cache = []

    def forward(self, x):
        res = self.sigmoid(x)
        self.last_call_cache.append(res)
        return res

    def backward(self, grad: np.ndarray):
        assert self.last_call_cache
        cache = self.last_call_cache.pop()
        return cache * (1. - cache) * grad.reshape(cache.shape)

    @staticmethod
    def sigmoid(x, slope=1.0):
        return 1.0 / (1.0 + np.exp(-slope * x))

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_ANY],


class IdFunction(Layer):
    def forward(self, x):
        return x

    def backward(self, grad: np.ndarray):
        return grad

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []

    def input_shape(self):
        return [SHAPE_BATCH, SHAPE_ANY],


class Dropout(Layer):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.cache = []

    def forward(self, x):
        res = x
        if self.is_train:
            mask = (1 + self.p) * np.random.binomial(n=2, p=1 - self.p, size=x.shape)
            self.cache.append(mask)
            res = res * mask
        return res

    def backward(self, dx):
        assert self.cache
        return dx * self.cache.pop()

    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def grads(self):
        return []


def softmax(x):
    emax = np.amax(x, axis=1, keepdims=True)
    exp = np.exp(x - emax)
    res = exp / np.sum(exp, axis=1, keepdims=True)
    return res


class CELoss:
    def __init__(self):
        self.cache = []

    def forward(self, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert len(y_pred.shape) == 2
        y_pred = np.clip(y_pred, eps, 1 - eps)
        self.cache.append({
            'y': y, 'y_pred': y_pred
        })

        res = y * np.log(y_pred)
        res = -np.sum(res, axis=1, keepdims=True)
        return res.mean()

    def backward(self) -> np.ndarray:
        assert self.cache, 'Cannot run backward pass before forward pass.'

        cache = self.cache.pop()
        return -(cache['y'] / cache['y_pred']) / cache['y_pred'].shape[0]


class CELossWithSoftmax:

    def __init__(self):
        self.cache = []

    def forward(self, y_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert len(y_raw.shape) == 2
        y_pred = softmax(y_raw)
        self.cache.append({
            'x': y_raw, 'y': y, 'y_pred': y_pred
        })

        res = y * np.log(np.clip(y_pred, eps, 1 - eps))
        res = -np.sum(res, axis=1, keepdims=True)
        return res.mean()

    def backward(self) -> np.ndarray:
        assert self.cache, 'Cannot run backward pass before forward pass.'

        cache = self.cache.pop()
        return (cache['y_pred'] - cache['y']) / cache['y_pred'].shape[0]


class CESequenceLoss:
    def __init__(self, with_softmax: bool = True):
        self.flatten_x = FlattenBatchLayer()
        self.flatten_y = FlattenBatchLayer()
        self.loss = CELossWithSoftmax() if with_softmax else CELoss()

    def forward(self, y_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_raw = self.flatten_x.forward(y_raw)
        y = self.flatten_y.forward(y)
        return self.loss.forward(y_raw, y)

    def backward(self) -> np.ndarray:
        dx = self.loss.backward()
        return self.flatten_x.backward(dx)


class NetWithLoss(Layer):
    def __init__(self, net, loss):
        super().__init__()
        self.net = net
        self.loss = loss

    def forward(self, x, y):
        res = self.net.forward(x)
        if type(res) == tuple:
            res = res[0]
        loss = self.loss.forward(res, y)
        return res, loss

    def backward(self):
        dx = self.loss.backward()
        dx = self.net.backward(dx)
        return dx

    def parameters(self):
        return self.net.parameters()

    def named_parameters(self):
        return self.net.named_parameters()

    def grads(self):
        return self.net.grads()

    def zerograd(self):
        for grad in self.grads():
            grad[:] = np.zeros_like(grad)

    def input_shape(self):
        return self.net.input_shape()
