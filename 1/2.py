import torch

x = torch.tensor(
    [[5., 10.], [1., 2.]], requires_grad=True)


def function(variable):
    return torch.prod(torch.log(torch.log(x + 7)))


def make_gradient_step(fun, variable):
    function_result = fun(variable)
    function_result.backward()
    print(variable, '<- variable')
    variable.data -= 0.001 * variable.grad
    variable.grad.zero_()


for i in range(500):
    make_gradient_step(function, x)


print("Значение x после 500 итераций с использованием SGD:")
print(x)