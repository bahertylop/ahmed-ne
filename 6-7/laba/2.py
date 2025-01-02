import torch

# Создаем входной тензор x
x = torch.tensor([[10., 20., 30.]])

# Оригинальный fc-слой с заданными весами и смещениями
fc = torch.nn.Linear(3, 3)
w = torch.tensor([[11., 12., 13.], [21., 22., 23.], [31., 32., 33.]])
b = torch.tensor([31., 32., 33.])
fc.weight.data = w
fc.bias.data = b

# Вычисляем выход оригинального fc-слоя
fc_out = fc(x)

# Реализуем эквивалент с матричным перемножением
my_fc_out = torch.mm(x, w.t()) + b

# Просуммируем выход fc-слоя, чтобы получить скаляр:
fc_out_summed = fc_out.sum()

# Посчитаем градиенты формулы fc_out_summed:
fc_out_summed.backward()
weight_grad = fc.weight.grad
bias_grad = fc.bias.grad

# Теперь воспроизведем вычисления выше но без fc-слоя:
# Проставим, что у "w" и "b" нужно вычислять градиенты (для fc-слоя это произошло автоматически):
w.requires_grad_(True)
b.requires_grad_(True)

# Получим выход нашей формулы:
our_formula = torch.mm(x, w.t()) + b
our_formula_summed = our_formula.sum()

# Сделайте backward для нашей формулы:
our_formula_summed.backward()

print("Градиенты весов совпадают:", torch.allclose(weight_grad, w.grad))
print("Градиенты смещений совпадают:", torch.allclose(bias_grad, b.grad))

# Проверка осуществляется автоматически, вызовом функций:
print('fc_weight_grad:', weight_grad)
print('our_weight_grad:', w.grad)
print('fc_bias_grad:', bias_grad)
print('out_bias_grad:', b.grad)
