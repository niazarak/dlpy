import itertools
from typing import NamedTuple

import numpy as np
import tqdm

from layers import LayerList, LSTMSampler, CESequenceLoss
from optim import Adam
from utils import one_hot, fix_seed


def generate_seq2seq_dataset(train_size: int, classes: int, seq_len: int, train: bool = True, task_sorting: bool = False):
    xs = []
    ys = []

    for t in range(train_size):
        x = np.random.randint(low=1, high=classes, size=seq_len)
        if task_sorting:
            # Sorting.
            res = sorted(x.tolist())
        else:
            # Reversing.
            res = x.tolist()[::-1]
        y = np.array([0] + res)
        x = one_hot(x, classes)
        y = one_hot(y, classes)
        xs.append(x)
        ys.append(y)
    xs = np.stack(xs)

    ys = np.stack(ys)
    if train:
        xs2, ys2 = ys[:, :seq_len].copy(), ys[:, 1:].copy()
    else:
        ys2 = ys[:, 1:]
        xs2 = np.zeros_like(ys2)

    return xs, xs2, ys2


class TrainArgs(NamedTuple):
    lr: float = 0.03
    batch_size: int = 4

    hidden_size: int = 12

    classes: int = 9
    seq_len: int = 4


def train_seq2seq(args: TrainArgs):
    lr = args.lr
    batch_size = args.batch_size
    classes = args.classes
    seq_len = args.seq_len
    hidden_size = args.hidden_size

    # DATA
    sorting = True
    fix_seed(1)
    xs, xs2, ys = generate_seq2seq_dataset(200, classes, seq_len, task_sorting=sorting)
    fix_seed(1)
    val_xs, val_xs2, val_ys = generate_seq2seq_dataset(20, classes, seq_len, task_sorting=sorting, train=False)

    train_size = xs.shape[0]
    batches_count = (train_size // batch_size) + int(train_size % batch_size > 0)

    # MODEL
    net = LayerList([
        LSTMSampler(
            hidden_size=hidden_size,
            encoder_input_features=classes,
            decoder_input_features=classes,
            teacher_forcing=False,
        )
    ])
    loss_mod = CESequenceLoss(with_softmax=False)
    optimizer = Adam(lr, net.parameters(), net.grads(), betas=(0.9, 0.999), eps=1e-8)

    best_val_loss = float('inf')
    for e in tqdm.tqdm(
            range(100), desc="Epoch",
    ):
        net.train()
        total_loss = 0
        for step in range(batches_count):
            batch_slice = slice(batch_size * step, batch_size * (step + 1))
            batch_xs, batch_xs2, batch_ys = xs[batch_slice], xs2[batch_slice], ys[batch_slice]
            batch_items = batch_xs.shape[0]

            batch_y_pred = net.forward(batch_xs, batch_xs2)[0]
            loss = loss_mod.forward(batch_y_pred, batch_ys)

            net.backward(loss_mod.backward())
            optimizer.step()

            optimizer.zerograd()

            total_loss += loss * batch_items

        total_loss /= train_size

        net.eval()
        val_y_pred = net.forward(val_xs, val_xs2)[0]
        val_loss = loss_mod.forward(val_y_pred, val_ys)

        best_val_loss = min(best_val_loss, val_loss)

        print('Epoch:', e, ', loss:', total_loss, ', val_loss:', val_loss)

    return best_val_loss


def grid_search_seq2seq():
    arg_space = [
        ('lr', [0.03, 0.01, 0.005, 0.003, 0.001]),
        ('batch_size', [1, 4, 8, 16, 32]),
        ('hidden_size', [4, 8, 16, 24]),
    ]
    arg_names = [name for name, _ in arg_space]
    for arg_set in itertools.product(*[space for _, space in arg_space]):
        args = TrainArgs(**dict(zip(arg_names, arg_set)))
        print("Eval with args:", args)
        loss = train_seq2seq(args)
        print("Loss:", loss)


if __name__ == '__main__':
    train_seq2seq(TrainArgs())
    # grid_search_seq2seq()
