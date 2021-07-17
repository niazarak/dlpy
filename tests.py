import unittest
from typing import Optional, Any

import numpy as np

from grad import L_gen, L_grad_gen, grad_check, copy_named_params
from layers import CELossWithSoftmax, SHAPE_BATCH, SHAPE_SEQ_LEN, Linear, LayerList, TanhFunction, SHAPE_ANY, ReLUFunction, SigmoidFunction, \
    IdFunction, LSTMGate, SimpleLSTM, SliceLayer, FlattenBatchLayer, CESequenceLoss, LSTMSampler, SoftmaxFunction, CELoss, Dropout
from utils import one_hot


def make_input_shape_for_net(net, batch_size, seq_len, any_size):
    shapes = []
    for inp in net.input_shape():
        shape = []
        for dim in inp:
            if dim == SHAPE_BATCH:
                shape.append(batch_size)
            elif dim == SHAPE_SEQ_LEN:
                shape.append(seq_len)
            elif dim == SHAPE_ANY:
                shape.append(any_size)
            else:
                shape.append(dim)
        shapes.append(shape)
    return shapes


class LayerInternalsTestCase(unittest.TestCase):
    def test_train_switch(self):
        dropout1 = Dropout(0.1)
        dropout2 = Dropout(0.1)
        linear = Linear(10, 10)
        lstm = LSTMSampler(10, 10, 10)
        layers = LayerList([
            dropout1,
            LayerList([
                dropout2,
                linear,
                lstm
            ])
        ])

        mods = [
            dropout1, dropout2, linear,
            lstm, lstm.encoder_cell, lstm.decoder_cell, lstm.encoder_cell.output_gate,
            lstm.encoder_cell.output_gate.activation
        ]

        layers.eval()
        assert all([not m.is_train for m in mods])

        layers.train()
        assert all([m.is_train for m in mods])


class GradCheckTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 20
        self.seq_len = 10
        self.any_size = 10

        self.loss_function = CELossWithSoftmax()

    def test_linear(self):
        out_size = 2
        net = Linear(5, out_size)
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, self.any_size)

    def test_tanh(self):
        out_size = 2
        net = TanhFunction()
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size)

    def test_relu(self):
        out_size = 2
        net = ReLUFunction()
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size)

    def test_sigmoid(self):
        out_size = 2
        net = SigmoidFunction()
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size)

    def test_id(self):
        out_size = 10
        net = IdFunction()
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size)

    def test_dropout(self):
        out_size = 10
        net = LayerList([
            IdFunction(),
            Dropout(0.1)
        ])
        net.train()
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size)

    def test_flatten(self):
        out_size = 10
        net = IdFunction()
        loss = CESequenceLoss()
        # self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size, y_seq_len=self.seq_len, loss_mod=loss)

    def test_net_1(self):
        out_size = 2
        net = LayerList(layers=[
            Linear(10, 5),
            TanhFunction(),
            Linear(5, out_size),
        ])

        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, self.any_size)

    def test_lstm_gate(self):
        out_size = 5
        net = LSTMGate(out_size, 10, TanhFunction())
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, self.any_size)

    def test_lstm_cell(self):
        out_size = 2
        seq_len = 5
        batch_size = 1
        net = LayerList([
            SimpleLSTM(out_size, 10),
            SliceLayer(np.s_[:, -1, :]),
        ])

        self.grad_check_net(
            net, batch_size, out_size, seq_len, self.any_size,
            eps=1e-06
        )

    def test_lstm_cell_seq(self):
        out_size = 2
        seq_len = 5
        batch_size = 1
        net = LayerList([
            SimpleLSTM(out_size, 10),
            FlattenBatchLayer()
        ])

        loss_mod = CESequenceLoss()

        self.grad_check_net(
            net, batch_size, out_size, seq_len, self.any_size,
            y_seq_len=seq_len, loss_mod=loss_mod, eps=1e-06
        )

    def test_lstm_seq2seq_teacher_forcing(self):
        feat_size = 6
        seq_len = 4
        batch_size = 3
        net = LayerList([
            LSTMSampler(feat_size, 10, feat_size, teacher_forcing=True),
        ])

        self.grad_check_net(
            net, batch_size, feat_size, seq_len, self.any_size,
            y_seq_len=seq_len, loss_mod=CESequenceLoss(with_softmax=False), eps=1e-06,
            forward_y=True,
        )

    def test_lstm_seq2seq(self):
        feat_size = 5
        seq_len = 5
        batch_size = 3
        net = LayerList([
            LSTMSampler(feat_size, 10, feat_size, teacher_forcing=False),
        ])

        self.grad_check_net(
            net, batch_size, feat_size, seq_len, self.any_size,
            y_seq_len=seq_len, loss_mod=CESequenceLoss(with_softmax=False), eps=1e-06,
            forward_y=True,
        )

    def test_softmax(self):
        out_size = 10
        net = SoftmaxFunction()
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size)

    def test_plain_ce(self):
        out_size = 10
        net = SoftmaxFunction()
        loss = CELoss()
        self.grad_check_net(net, self.batch_size, out_size, self.seq_len, out_size, loss_mod=loss)

    def test_ce_softmax_decomposition(self):
        input_size = 10
        batch_size = 5

        softmax = SoftmaxFunction()
        loss = CELoss()

        softmax_loss = CELossWithSoftmax()

        x = np.random.normal(size=(batch_size, input_size))
        y = one_hot(np.random.randint(0, input_size, batch_size), input_size)

        assert (loss.forward(softmax.forward(x), y) == softmax_loss.forward(x, y)).all()
        assert np.allclose(softmax.backward(loss.backward()), softmax_loss.backward())

    def grad_check_net(
            self, net,
            x_batch_size, out_size, seq_len, any_size,
            y_seq_len=None, eps=1e-5, tol=1e-4,
            loss_mod: Optional[Any] = None, forward_y: bool = False,
    ):
        y_batch_size = x_batch_size
        if y_seq_len is not None:
            y_batch_size *= y_seq_len
        y = one_hot(np.random.randint(0, out_size, y_batch_size), out_size)
        if y_seq_len is not None:
            y_batch_size //= y_seq_len
            y = y.reshape(y_batch_size, y_seq_len, out_size)

        input_shapes = make_input_shape_for_net(net=net, batch_size=x_batch_size, seq_len=seq_len, any_size=any_size)
        x = [np.random.random(size=shape) for shape in input_shapes]
        if forward_y:
            x.append(y)
        w = copy_named_params(net.named_parameters())
        for name, param in w:
            param[:] = np.random.normal(size=param.shape)
            # param[:] = np.zeros_like(param)
        w = [(f'input.{i}', x_i) for i, x_i in enumerate(x)] + w

        if not loss_mod:
            loss_mod = self.loss_function

        J = lambda _w: L_gen(net=net, loss_mod=loss_mod, ws=_w[len(x):], x=_w[:len(x)], y=y)
        grad_J = lambda _w: L_grad_gen(net=net, loss_mod=loss_mod, ws=_w[len(x):], x=_w[:len(x)], y=y)

        grad_check(J, grad_J, w, eps=eps, tol=tol)


if __name__ == '__main__':
    unittest.main()
