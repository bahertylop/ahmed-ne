import torch
import matplotlib.pyplot as plt

# Задаем функцию для предсказаний и построения графика
def predict(net, x, y):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', label='Валидный набор')  # Истинные значения
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Нейросетка')  # Предсказания
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Нейросетка vs Валидный набор')
    plt.show()

# Задаем нашу целевую функцию
def target_function(x):
    return 3**x * torch.sin(x + 2)

# Задаем функцию для расчета метрики (MAE)
def metric(pred, target):
    return (pred - target).abs().mean()

# Определяем архитектуру нейронной сети
class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)  # Первый слой
        self.act1 = torch.nn.Sigmoid()                     # Активация
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1) # Второй слой (выходной)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

# Создаем модель
net = RegressionNet(n_hidden_neurons=20)  # 10 нейронов в скрытом слое

# ------Подготовка датасета--------:
x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.  # Добавляем шум
y_train = y_train + noise
x_train.unsqueeze_(1)  # Добавляем размерность
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Подготовка датасета завершена--------:

# Оптимизатор
optimizer = torch.optim.Adam(net.sq, lr=0.01)  # Шаг обучения

# Функция потерь (MSE)
def loss(pred, target):
    return ((pred - target) ** 2).mean()

# Обучение модели
epochs = 10000  # Количество эпох

for epoch_index in range(epochs):
    optimizer.zero_grad()  # Обнуляем градиенты

    y_pred = net.forward(x_train)  # Прямой проход
    loss_value = loss(y_pred, y_train)  # Вычисление потерь

    loss_value.backward()  # Обратный проход
    optimizer.step()  # Шаг оптимизации

    # Печать ошибки каждые 200 эпох
    if epoch_index % 200 == 0:
        print(f"Epoch {epoch_index}: Loss = {loss_value.item():.4f}")

# Выводим метрику MAE:
mae = metric(net.forward(x_validation), y_validation).item()
print(f"Validation MAE: {mae:.4f}")

# Строим график:
predict(net, x_validation, y_validation)
