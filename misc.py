import numpy as np
import torch.nn

from layers import CELossWithSoftmax, LayerList, NetWithLoss, Linear
from optim import Adam


def compare_with_torch():
    my_linear = Linear(2, 2)
    my_loss = CELossWithSoftmax()
    net = NetWithLoss(
        net=LayerList([
            # SimpleLSTM(2, 2),
            # SliceLayer(np.s_[:, -1, :]),
            my_linear
        ]),
        loss=my_loss
    )

    torch_layer = torch.nn.Linear(2, 2)
    # torch_layer = torch.nn.LSTM(2, 2, 1)
    torch_layer.weight = torch.nn.Parameter(torch.tensor(my_linear.weight.T))
    torch_layer.bias = torch.nn.Parameter(torch.tensor(my_linear.bias))
    torch_loss = torch.nn.CrossEntropyLoss()

    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    optimizer = Adam(lr, net.parameters(), net.grads(), betas=betas, eps=eps)
    torch_opt = torch.optim.Adam(torch_layer.parameters(), lr, betas=betas, eps=eps)

    x = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    y = np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.float)

    # x, y = make_lstm_input(0, 4, 2)
    for i in range(2000):
        yh, loss = net.forward(x, y)
        # my_out = my_linear.forward(x)
        # loss = my_loss.forward(my_out, y)

        torch_out = torch_layer.forward(torch.tensor(x))
        tloss = torch_loss.forward(torch_out, torch.tensor(y).argmax(dim=1))
        # torch_out = torch_layer.forward(torch.tensor(x.transpose((1, 0, 2))).float())
        # tloss = torch_loss.forward(torch_out[0][-1, :, :], torch.tensor(y).argmax(dim=1))

        print("Loss", loss),
        print("Loss (torch)", tloss),
        # import pdb; pdb.set_trace()

        # MY
        net.backward()
        # optimizer.step(yh.shape[0])
        optimizer.step()

        net.zerograd()

        #     TORCH
        tloss.backward()
        torch_opt.step()
        torch_opt.zero_grad()
