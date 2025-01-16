import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Установка начального состояния для всех генераторов случайных чисел
# Это нужно для воспроизводимости результатов
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
# рассказать за сиды внутри


# Загрузка датасета "wine" из sklearn
wine = load_wine()
features = 13  # Используем все 13 признаков из набора данных

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


unique, counts = np.unique(wine.target, return_counts=True)
base_rate = max(counts) / sum(counts)
print("Base Rate:", base_rate)

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
test_results = {}

for ts in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=ts, shuffle=True
    )
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Параметры нейронной сети
    n_input = 13  # Количество входных нейронов (признаки)
    n_hidden = 10  # Количество нейронов в скрытом слое
    wine_net = WineNet(n_input, n_hidden)  # Создаем экземпляр сети

    # Функция потерь (кросс-энтропия для многоклассовой классификации)
    loss = torch.nn.CrossEntropyLoss()
    # Оптимизатор Adam
    optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-3)
    results1 = []

    batch_size = 10  # Размер батча для обучения
    for epoch in range(500):  # Количество эпох
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
        if epoch > 490:
            test_preds1 = wine_net.forward(X_test).argmax(dim=1)
            accuracy1 = (test_preds1 == y_test).float().mean().item()
            results1.append(accuracy1)

    test_preds = wine_net.forward(X_test).argmax(dim=1)
    accuracy = (test_preds == y_test).float().mean().item()
    test_results[ts] = accuracy
    print(ts, results1)

print("Точность для разных размеров тестовой части датасета:", test_results)
# Результат:
# Base Rate: 0.398876404494382
# 0.1 0.8888888955116272
# 0.2 0.8888888955116272
# 0.3 0.9814814925193787
# 0.4 0.6944444179534912
# 0.5 0.8314606547355652
# 0.6 0.8504672646522522
# 0.7 0.8799999952316284
# 0.8 0.8601398468017578
# 0.9 0.6708074808120728

# Base Rate: 0.398876404494382
# 0.1 [0.6666666865348816, 0.6666666865348816, 0.7222222089767456,
#      0.6666666865348816, 0.7222222089767456, 0.7222222089767456,
#      0.6666666865348816, 0.7222222089767456, 0.7222222089767456]
# 0.2 [0.8888888955116272, 0.9166666865348816, 0.9444444179534912,
#      0.9166666865348816, 0.9166666865348816, 0.8888888955116272,
#      0.9444444179534912, 0.9166666865348816, 0.9166666865348816]
# 0.3 [0.4444444477558136, 0.4444444477558136, 0.4444444477558136,
#      0.4444444477558136, 0.4444444477558136, 0.4444444477558136,
#      0.4444444477558136, 0.4444444477558136, 0.4444444477558136]
# 0.4 [0.4166666567325592, 0.4166666567325592, 0.4166666567325592,
#      0.4166666567325592, 0.4166666567325592, 0.4166666567325592,
#      0.4166666567325592, 0.4166666567325592, 0.4166666567325592]
# 0.5 [0.3820224702358246, 0.3820224702358246, 0.3820224702358246,
#      0.3820224702358246, 0.3820224702358246, 0.3820224702358246,
#      0.3820224702358246, 0.3820224702358246, 0.3820224702358246]
# 0.6 [0.8785046935081482, 0.8971962332725525, 0.8878504633903503,
#      0.8878504633903503, 0.8971962332725525, 0.8971962332725525,
#      0.8878504633903503, 0.8878504633903503, 0.8971962332725525]
# 0.7 [0.7919999957084656, 0.7919999957084656, 0.8080000281333923,
#      0.8080000281333923, 0.800000011920929, 0.8080000281333923,
#      0.7919999957084656, 0.800000011920929, 0.8080000281333923]
# 0.8 [0.3636363744735718, 0.3636363744735718, 0.3636363744735718,
#      0.3636363744735718, 0.3636363744735718, 0.3636363744735718,
#      0.3636363744735718, 0.3636363744735718, 0.3636363744735718]
# 0.9 [0.6708074808120728, 0.6708074808120728, 0.6708074808120728,
#      0.6770186424255371, 0.6770186424255371, 0.6770186424255371,
#      0.6770186424255371, 0.6770186424255371, 0.6770186424255371]
