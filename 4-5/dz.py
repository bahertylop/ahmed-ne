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

# Загрузка датасета "wine" из sklearn
wine = load_wine()
features = 13  # Используем все 13 признаков из набора данных

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, :features],  # используем все признаки
    wine.target,              # метки классов
    test_size=0.3,            # 30% данных пойдут в тестовую выборку
    shuffle=True)             # перемешиваем данные перед разделением

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

    # Метод для получения предсказаний (с Softmax)
    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


# Параметры нейронной сети
n_input = 13  # Количество входных нейронов (признаки)
n_hidden = 20  # Количество нейронов в скрытом слое
wine_net = WineNet(n_input, n_hidden)  # Создаем экземпляр сети

# Функция потерь (кросс-энтропия для многоклассовой классификации)
loss = torch.nn.CrossEntropyLoss()
# Оптимизатор Adam
optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-3)

batch_size = 5  # Размер батча для обучения

# Обучение сети
for epoch in range(1000):  # Количество эпох
    # Перемешиваем индексы обучающей выборки
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()  # Обнуляем градиенты перед каждым шагом

        # Получаем индексы для текущего батча
        batch_indexes = order[start_index:start_index + batch_size]

        # Извлекаем батч данных
        x_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        # Прямой проход
        preds = wine_net.forward(x_batch)

        # Вычисляем значение функции потерь
        loss_value = loss(preds, y_batch)
        loss_value.backward()  # Обратное распространение ошибки

        optimizer.step()  # Обновляем веса сети

    # Каждые 100 эпох вычисляем точность на тестовой выборке
    if epoch % 100 == 0:
        test_preds = wine_net.forward(X_test)
        test_preds = test_preds.argmax(dim=1)  # Получаем предсказанные классы
        print((test_preds == y_test).float().mean())  # Выводим точность

# Проверка количества входных признаков и финальной точности
print(wine_net.fc1.in_features, np.asarray((test_preds == y_test).float().mean()) > 0.8)
# Нужно получить "13 True" - это значит, что сеть правильно обучается на 13 признаках
