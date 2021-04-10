import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from numpytorch.nn import Dense, Sequential
from numpytorch.activations import ReLU, Sigmoid
from numpytorch import losses
from numpytorch import optim

iris = pd.read_csv("Iris.csv")
iris = iris.sample(frac=1).reset_index(drop=True)  # Shuffle
X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
X = np.array(X)
one_hot_encoder = OneHotEncoder(sparse=False)

Y = iris.Species
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

model = Sequential(
    Dense(4, 5, ReLU, xavier_init=True),
    Dense(5, 10, ReLU, xavier_init=True),
    Dense(10, 3, Sigmoid),
)
optimizer = optim.SGD(0.01, model.parameters())
loss_fun = losses.BinaryCrossEntropy()

steps = 1000
t = tqdm(total=steps)
for _ in range(steps):
    optimizer.zero_grad()
    a = model(X_train)

    loss = loss_fun(y_train, a)

    dA = loss_fun.backward(y_train, a)
    model.backward(dA)

    optimizer.step()
    t.set_postfix(
        loss=loss.mean(),
        w_mean=model.modules[0].w.data.mean(),
        dw_mean=model.modules[0].w.grad.mean(),
    )
    t.update()
t.close()

a = model(X_train)
y_pred = np.zeros_like(a)
inds = np.argmax(a, axis=1)
for i, j in zip(range(y_pred.shape[0]), inds):
    y_pred[i, j] = 1
print("Train accuracy: ", (y_pred == y_train).all(axis=1).sum() / y_train.shape[0])

a = model(X_test)
y_pred = np.zeros_like(a)
inds = np.argmax(a, axis=1)
for i, j in zip(range(y_pred.shape[0]), inds):
    y_pred[i, j] = 1
print("Test accuracy: ", (y_pred == y_test).all(axis=1).sum() / y_test.shape[0])
