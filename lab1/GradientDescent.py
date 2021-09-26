from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def sigmoid_activation(x):

    return  1.0 / (1+ np.exp(-x))

def sigmoid_deriv(x):
    return x * (1-x)

def predict(X,W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds <=0.5] = 0
    preds[preds > 0.5] = 1
    return preds

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)

y = y.reshape((y.shape[0],1))

X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

W = np.random.randn(X.shape[1], 1)
losses = []

alpha = 0.01
epochs = 100

for epoch in np.arange(0, epochs):

    preds = sigmoid_activation(trainX.dot(W))
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)

    W += -alpha * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("Epoch = {}, loss={:.7f}".format(int(epoch+1), loss))

print("Evaluating the Model")
preds = predict(testX,W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:,0], testX[:,1], marker="o", c=testY[:,0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
