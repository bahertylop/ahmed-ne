import torch
import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt

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

len(y_train), len(y_test)

plt.imshow(X_train[0, :, :])
plt.show()
print(y_train[0])

X_train = X_train.unsqueeze(1).float()
X_test = X_test.unsqueeze(1).float()

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


lenet5 = LeNet5()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# lenet5 = lenet5.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lenet5.parameters(), lr=1.0e-3)
# optimizer = torch.optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)

batch_size = 512

test_accuracy_history = []
test_loss_history = []

# X_test = X_test.to(device)
# y_test = y_test.to(device)

for epoch in range(100):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes]  # .to(device)
        y_batch = y_train[batch_indexes]  # .to(device)

        preds = lenet5.forward(X_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    test_preds = lenet5.forward(X_test)
    test_loss_history.append(loss(test_preds, y_test).data.cpu())

    accuracy = ((test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()).item()
    test_accuracy_history.append(accuracy)

    print(accuracy)

lenet5.forward(X_test)

# Отображение графика точности
plt.figure(figsize=(10, 5))
plt.plot(test_accuracy_history, label='Точность')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid()
plt.show()

# Отображение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(test_loss_history, label='Потери', color='red')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.grid()
plt.show()

# tensor(5)
# 0.974299967288971
# 0.9835000038146973
# 0.9842999577522278
# 0.9891999959945679
# 0.9892999529838562
# 0.9886999726295471
# 0.9898999929428101
# 0.9904999732971191
# 0.9918999671936035
# 0.9889999628067017
# 0.9922999739646912
# 0.9914000034332275
# 0.9919999837875366
# 0.9918999671936035
# 0.9916999936103821
# 0.9906999468803406
# 0.9899999499320984
# 0.9908999800682068
# 0.9905999898910522
# 0.9911999702453613
# 0.990399956703186
# 0.9887999892234802
# 0.991599977016449
# 0.9891999959945679
# 0.9902999997138977
# 0.990399956703186
# 0.9905999898910522
# 0.9917999505996704
# 0.9922999739646912
# 0.9926999807357788
# 0.9925999641418457
# 0.9924999475479126
# 0.9924999475479126
# 0.9925999641418457
# 0.9923999905586243
# 0.9922999739646912
# 0.9923999905586243
# 0.9923999905586243
# 0.9923999905586243
# 0.9925999641418457
# 0.9924999475479126
# 0.9924999475479126
# 0.9925999641418457
# 0.9925999641418457
# 0.9925999641418457
# 0.9926999807357788
# 0.9926999807357788
# 0.9924999475479126
# 0.9926999807357788
# 0.9926999807357788
# 0.9926999807357788
# 0.9924999475479126
# 0.9926999807357788
# 0.9926999807357788
# 0.9926999807357788
# 0.9925999641418457
# 0.9924999475479126
# 0.9924999475479126
# 0.9926999807357788
# 0.9926999807357788
# 0.9926999807357788
# 0.9924999475479126
# 0.9924999475479126
# 0.9926999807357788
# 0.9926999807357788
# 0.9926999807357788
# 0.9928999543190002
# 0.9928999543190002
# 0.9928999543190002
# 0.9927999973297119
# 0.9928999543190002
# 0.9928999543190002
# 0.9927999973297119
# 0.9928999543190002
# 0.9926999807357788
# 0.9927999973297119
# 0.9928999543190002
# 0.9927999973297119
# 0.9927999973297119
# 0.9927999973297119
# 0.9926999807357788
# 0.9925999641418457
# 0.9926999807357788
# 0.9926999807357788
# 0.9924999475479126
# 0.9925999641418457
# 0.9927999973297119
# 0.9927999973297119
# 0.9928999543190002
# 0.9926999807357788
# 0.9927999973297119
# 0.9926999807357788
# 0.9927999973297119
# 0.9927999973297119
# 0.9926999807357788
# 0.9925999641418457
# 0.9926999807357788
# 0.9925999641418457
# 0.9926999807357788
# 0.9927999973297119

# по графику видно, что после 30+- эпох, loss начинает расти, что приводит к переобучению

# по времени сверточные быстрее, но слоев больше, поэтому время примерно одинаковое