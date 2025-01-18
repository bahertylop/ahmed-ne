import torch

N = 4
C = 3
C_out = 10
H = 8
W = 16
x = torch.ones((N, C, H, W))

torch.Size([4, 10, 8, 16])
out1 = torch.nn.Conv2d(C, C_out, kernel_size=(3, 3), padding=1)(x)
print('Мой ответ:', out1.shape)  # для самопроверки
print("Ответ: ", torch.Size([4, 10, 8, 16]), '\n')

torch.Size([4, 10, 8, 16])
out2 = torch.nn.Conv2d(C, C_out, kernel_size=(5, 5), padding=2)(x)
print('Мой ответ:', out2.shape)  # для самопроверки
print("Ответ: ", torch.Size([4, 10, 8, 16]), '\n')

torch.Size([4, 10, 8, 16])
out3 = torch.nn.Conv2d(C, C_out, kernel_size=(7, 7), padding=3)(x)
print('Мой ответ:', out3.shape)  # для самопроверки
print("Ответ: ", torch.Size([4, 10, 8, 16]), '\n')

torch.Size([4, 10, 8, 16])
out4 = torch.nn.Conv2d(C, C_out, kernel_size=(9, 9), padding=4)(x)
print('Мой ответ:', out4.shape)  # для самопроверки
print("Ответ: ", torch.Size([4, 10, 8, 16]), '\n')

torch.Size([4, 10, 8, 16])
out5 = torch.nn.Conv2d(C, C_out, kernel_size=(3, 5), padding=(1, 2))(x)
print('Мой ответ:', out5.shape)  # для самопроверки
print("Ответ: ", torch.Size([4, 10, 8, 16]), '\n')

torch.Size([4, 10, 22, 30])
out6 = torch.nn.Conv2d(C, C_out, kernel_size=(3, 3), padding=8)(x)
print('Мой ответ:', out6.shape)  # для самопроверки
print('Ответ: ', torch.Size([4, 10, 22, 30]), '\n')

torch.Size([4, 10, 7, 15])
out7 = torch.nn.Conv2d(C, C_out, kernel_size=(4, 4), padding=1)(x)
print('Мой ответ:', out7.shape)  # для самопроверки
print('Ответ: ', torch.Size([4, 10, 7, 15]), '\n')

torch.Size([4, 10, 9, 17])
out8 = torch.nn.Conv2d(C, C_out, kernel_size=(2, 2), padding=1)(x)
print('Мой ответ:', out8.shape)  # для самопроверки
print('Ответ: ', torch.Size([4, 10, 9, 17]), '\n')
