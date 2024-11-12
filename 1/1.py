import torch

x = torch.tensor(
    [[5., 10.], [1., 2.]], requires_grad=True)

function = torch.prod(torch.log(torch.log(x + 7)))

function.backward()

print(x.grad, '<- gradient')