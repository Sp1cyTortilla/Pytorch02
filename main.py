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
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
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

y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

y_pred = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

print(torch.eq(y_pred.squeeze(), y_pred_labels.squeeze()))

y_pred.squeeze_()

print(y_pred[:5])