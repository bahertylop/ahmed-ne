import torch
import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt
import time

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

MNIST_train = torchvision.datasets.MNIST('../', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('../', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train.dtype, y_train.dtype

X_train = X_train.float()
X_test = X_test.float()

X_train.shape, X_test.shape

y_train.shape, y_test.shape

X_train = X_train.reshape([-1, 28 * 28])
X_test = X_test.reshape([-1, 28 * 28])


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


mnist_net = MNISTNet(100)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)  # adam

batch_size = 100

train_loss_history = []
validation_loss_history = []
epochs = 100

start = time.time()

for epoch in range(epochs):
    train_loss = 0.0
    order = np.random.permutation(len(X_train))

    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        preds = mnist_net(X_batch)
        loss_value = loss(preds, y_batch)
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

    train_loss_history.append(train_loss / (len(X_train) / batch_size))

    test_preds = mnist_net.forward(X_test)
    validation_loss = loss(test_preds, y_test)
    validation_loss_history.append(validation_loss.item())

    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    print(accuracy)

finish = time.time()
print("time = ", finish-start)

plt.plot(train_loss_history, label='Train Loss')
plt.plot(validation_loss_history, label='Validation Loss')
plt.legend()
plt.show()
