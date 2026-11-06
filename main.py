#pytorch workbook number 2


from sklearn.datasets import make_circles

n_samples = 1000

#create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

#print(f"X shape: {X[:5]}")
#print(f"y shape: {y[:5]}")

import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
#print(circles.head(10))
#print(circles.label.value_counts())

import matplotlib.pyplot as plt
#plt.scatter(x=X[:, 0],
            #y=X[:, 1],
            #c=y,
            #cmap=plt.cm.RdYlBu)
#plt.show()

#print(X.shape, y.shape)


import torch
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

#print(X[:5], y[:5])

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(len(X_train), len(X_test), len(y_train), len(y_test))

#creating port for CPU to work from
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

class Circlemodel(nn.Module):
    def __init__(self):
        super(Circlemodel, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x))
#model_0 = Circlemodel().to(device)
#print(model_0)
model_0 = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1)).to(device)
#print(model_0)


untrained_preds = model_0(X_test.to(device))
print(f"untrained predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of Test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n {untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n {y_test[:10]}")


#create a loss function
#loss function shows how wrong your model is
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

#create evaluation metric
#eval metric shows how right your model is

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_true)) * 100
    return acc

y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

#reminder that logits are just models raw outputs
#activation function
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

y_pred = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

print(torch.eq(y_pred.squeeze(), y_pred_labels.squeeze()))

y_pred.squeeze_()

print(y_test[:5])


#Training loop
torch.manual_seed(42)

epochs = 100

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss =  loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        0,
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {test_acc:.4f}")

#predictions

import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_functions.py exists")
else:
    print("downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open(Path("helper_functions.py"), "w") as f:
        f.write(request.content)
from helper_functions import plot_predictions, plot_decision_boundary

#plt.figure(figsize=(12, 6))
#plt.subplot(1,2,1)
#plt.title("train")
#plot_decision_boundary(model_0, X_train, y_train)
#plt.subplot(1,2,2)
#plt.title("test")
#lot_decision_boundary(model_0, X_test, y_test)
#plt.show()


class CircleModelV1(nn.Module):
    def __init__(self):
        super(CircleModelV1, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))

model_1 = CircleModelV1().to(device)
print(model_1)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)

torch.manual_seed(42)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):

    #forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))


    #Calculate loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    #optimizer
    optimizer.zero_grad()

    #loss
    loss.backward()

    #optimizer step
    optimizer.step()

    #test
    model_1.eval()
    with torch.inference_mode():

        #forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        #test acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, Loss: {loss:.5f}, Accuracy: {acc:.2f}, Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}")


#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title("Train")
#lot_decision_boundary(model_1, X_train, y_train)
#plt.subplot(1, 2, 2)
#plt.title("Test")
#plot_decision_boundary(model_1, X_test, y_test)
#plt.show()



weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight*X_regression + bias

print(len(X_regression))
print(X_regression[:5], y_regression[:5])

train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]
print(len(X_train_regression), len(y_train_regression), len(X_test_regression), len(y_test_regression))

#plot_predictions(train_data=X_train_regression,
#                train_labels=y_train_regression,
#                 test_data=X_test_regression,
#                 test_labels=y_test_regression,
#                 );
#plt.show()

model_2 = nn.Sequential(nn.Linear(1, 10), nn.Linear(10, 10), nn.Linear(10,1) ).to(device)
print(model_2)

#Loss and Optimizer

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)


#Train the model
torch.manual_seed(42)

epochs = 1000
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range(epochs):
    #pass forward
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)

        test_loss = loss_fn(test_pred, y_test_regression)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.5f}, Accuracy: {test_loss:.5f}")

model_2.eval()
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu())
plt.show()



