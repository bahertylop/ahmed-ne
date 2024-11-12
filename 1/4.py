import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)


def function(variable):
    return torch.prod(torch.log(torch.log(variable + 7)))


optimizer = torch.optim.SGD([x], lr=0.001)

fn_history_sgd = []

for i in range(500):
    optimizer.zero_grad()
    function_result = function(x)
    function_result.backward()
    optimizer.step()

    fn_history_sgd.append(function_result.item())

print("Значение x после 500 итераций с использованием SGD:")
print(x)


plt.plot(range(500), fn_history_sgd, label="График")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Реализация градиентного спуска")
plt.legend()
plt.show()