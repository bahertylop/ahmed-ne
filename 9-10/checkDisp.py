import torch
import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt


def learning_LeNet(model_class, batch_size_par, X_test_v, y_test_v, X_Train_v, Y_train):
    lenet5 = model_class()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lenet5 = lenet5.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lenet5.parameters(), lr=1.0e-3)
    # optimizer = torch.optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)

    batch_size = batch_size_par

    test_accuracy_history = []
    test_loss_history = []

    X_test = X_test_v.to(device)
    y_test = y_test_v.to(device)
    X_train = X_Train_v
    y_train = Y_train


    for epoch in range(50):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = lenet5.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = lenet5.forward(X_test)
        test_loss_history.append(loss(test_preds, y_test).data.cpu())

        accuracy = ((test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()).item()
        test_accuracy_history.append(accuracy)

        print(accuracy)

    return max(test_accuracy_history)


def learning_MNIST(model_class, batch_size_par, X_test_v, y_test_v, X_train_v, y_train_v):
    mnist_net = model_class(100)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mnist_net = mnist_net.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mnist_net.parameters(), lr=1.0e-3)

    batch_size = batch_size_par

    test_accuracy_history = []
    X_test = X_test_v.to(device)
    y_test = y_test_v.to(device)
    X_train = X_train_v
    y_train = y_train_v
    for epoch in range(50):
        order = np.random.permutation(len(X_train))

        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = mnist_net.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = mnist_net.forward(X_test)
        accuracy = ((test_preds.argmax(dim=1) == y_test).float().mean().cpu()).item()
        test_accuracy_history.append(accuracy)
        print(accuracy)
    return max(test_accuracy_history)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

plt.imshow(X_train[0, :, :])
plt.show()

X_train_LeNet = X_train.unsqueeze(1).float()
X_test_LeNet = X_test.unsqueeze(1).float()

X_train_MNIST = X_train.float().reshape([-1, 28 * 28])
X_test_MNIST = X_test.float().reshape([-1, 28 * 28])

in_channels = 1
conv1_out_channels = 8
conv2_out_channels = 64
kernel_size = 5
padding = 2
stride_pool = 2
pool_kernel_size = 2
fc1_out_features = 160
fc2_out_features = 100
num_classes = 10


class MNISTNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=conv1_out_channels, kernel_size=kernel_size, padding=padding)
        self.act1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=pool_kernel_size, stride=stride_pool)

        self.conv2 = torch.nn.Conv2d(
            in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=kernel_size, padding=0)
        self.act2 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=pool_kernel_size, stride=stride_pool)

        self.fc1 = torch.nn.Linear(5 * 5 * 64, fc1_out_features)
        self.act3 = torch.nn.Tanh()

        self.fc2 = torch.nn.Linear(fc1_out_features, fc2_out_features)
        self.act4 = torch.nn.Tanh()

        self.fc3 = torch.nn.Linear(fc2_out_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


accuracy_LeNet = []
accuracy_fc = []

for i in range(10):
    print(i, "fc")
    accuracy_fc.append(learning_MNIST(MNISTNet, 100, X_test_MNIST, y_test, X_train_MNIST, y_train))
    print(i, "LeNet")
    accuracy_LeNet.append(learning_LeNet(LeNet5, 512, X_test_LeNet, y_test, X_train_LeNet, y_train))

plt.figure(figsize=(10, 6))
plt.plot(accuracy_LeNet, label="LeNet5", marker='o')
plt.plot(accuracy_fc, label="Fully Connected", marker='s', linestyle='--')

# Оформление
plt.title("Сравнение точности LeNet5 и Fully Connected моделей")
plt.xlabel("Запуски")
plt.ylabel("Точность")
plt.legend()
plt.grid(True)
plt.show()