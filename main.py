
from torch.utils.data import DataLoader
import data
import NeuralNetwork as NN
import train as tn
from train import torch

train_dataloader = DataLoader(data.training_data, batch_size=data.batch_size)
test_dataloader = DataLoader(data.test_data, batch_size=data.batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NN.NeuralNetwork().to(device)

loss_fn = NN.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tn.train(train_dataloader, model, loss_fn, optimizer)
    tn.test(test_dataloader, model, loss_fn)
print("Done")
