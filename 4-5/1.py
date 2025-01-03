import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Установка начального состояния для всех генераторов случайных чисел
# Это нужно для воспроизводимости результатов
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
# рассказать за сиды внутри


# Загрузка датасета "wine" из sklearn
wine = load_wine()
features = 13  # Используем все 13 признаков из набора данных

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, :features],  # используем все признаки
    wine.target,  # метки классов
    test_size=0.3,  # 30% данных пойдут в тестовую выборку
    shuffle=True)  # перемешиваем данные перед разделением

# Конвертация данных в тензоры PyTorch
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# Определение нейронной сети
class WineNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden_neurons):
        super(WineNet, self).__init__()
        # Первый полносвязный слой (входной)
        self.fc1 = torch.nn.Linear(n_input, n_hidden_neurons)
        self.activ1 = torch.nn.Sigmoid()  # Сигмоидальная функция активации
        # Второй полносвязный слой
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.activ2 = torch.nn.Sigmoid()  # Еще одна сигмоидальная активация
        # Выходной слой (3 класса)
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        # Softmax для получения вероятностей классов
        self.sm = torch.nn.Softmax(dim=1)

    # Прямой проход (forward) сети
    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        return x

    #  вторая причина не применения софтаммакса
    # Метод для получения предсказаний (с Softmax)
    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


hidden_neurons = [1, 2, 5, 10, 12, 13, 14, 20, 50, 100]
results = {}

batch_size = 10

for n_hidden in hidden_neurons:
    wine_net = WineNet(n_input=13, n_hidden_neurons=n_hidden)
    optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-3)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(1000):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            batch_indexes = order[start_index:start_index + batch_size]
            x_batch = X_train[batch_indexes]
            y_batch = y_train[batch_indexes]

            preds = wine_net.forward(x_batch)
            loss_value = loss(preds, y_batch)
            loss_value.backward()
            optimizer.step()

    test_preds = wine_net.forward(X_test).argmax(dim=1)
    accuracy = (test_preds == y_test).float().mean().item()
    results[n_hidden] = accuracy
    print(n_hidden, accuracy)

print("Точность для различного количества скрытых нейронов:", results)

# Результат:
# 1 0.40740740299224854
# 2 0.40740740299224854
# 5 0.40740740299224854
# 10 0.40740740299224854
# 12 0.40740740299224854
# 13 0.9074074029922485
# 14 0.9259259104728699
# 20 0.9629629850387573
# 50 0.9629629850387573
# 100 0.9629629850387573
# вывод: нейросеть обучается только, если количество скрытых нейронов >= количеству нейронов на входе


# линейное разделение при 1
# количество данных в датасете большое