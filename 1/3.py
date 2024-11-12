import torch

x = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)


def function(variable):
    return torch.prod(torch.log(torch.log(variable + 7)))


optimizer = torch.optim.SGD([x], lr=0.001)


for i in range(500):
    optimizer.zero_grad()
    function_result = function(x)
    function_result.backward()
    optimizer.step()


print("Значение x после 500 итераций с использованием SGD:")
print(x)

