# simple playground to test weights loading in torch models
import torch
import random


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear_1 = torch.nn.Linear(10, 10)
        self.linear_2 = torch.nn.Linear(10, 10)
        self.linear_3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x

model = MyModel()

# quick train loop on dummy data
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(2):
    for _ in range(5):
        inputs = torch.randn(4, 10)
        targets = torch.randn(4, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# save weights
torch.save(model.state_dict(), "simple_weights.pth")

model.modules

class MyModelExtraLayer(torch.nn.Module):
    def __init__(self):
        super(MyModelExtraLayer, self).__init__()
        self.linear_1 = torch.nn.Linear(10, 10)
        self.linear_2 = torch.nn.Linear(10, 10)
        self.extra_layer = torch.nn.Linear(10, 10)
        self.linear_3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.extra_layer(x)
        x = self.linear_3(x)
        return x

model_extra = MyModelExtraLayer()

# load weights
state = torch.load("simple_weights.pth")
missing, unexpected = model_extra.load_state_dict(state, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
