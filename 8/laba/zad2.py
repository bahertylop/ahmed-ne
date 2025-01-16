import numpy as np


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    # Извлекаем входные параметры
    batch_size, in_channels, in_height, in_width = input_matrix_shape

    # Вычисляем размеры выхода для высоты и ширины
    out_height = (in_height - kernel_size + 2 * padding) // stride + 1
    out_width = (in_width - kernel_size + 2 * padding) // stride + 1

    # Возвращаем итоговую форму выходной матрицы
    return [batch_size, out_channels, out_height, out_width]

print(np.array_equal(
    calc_out_shape(input_matrix_shape=[2, 3, 10, 10],
                   out_channels=10,
                   kernel_size=3,
                   stride=1,
                   padding=0),
    [2, 10, 8, 8]))

