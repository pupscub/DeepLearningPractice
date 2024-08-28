import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt                                    


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.cv1 = nn.Conv2d(1,16,3,1)
        self.cv2 = nn.Conv2d(16,32,3,1)
        self.dp1 = nn.Dropout(0.25)
        self.dp2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self,x):
        x = self.cv1(x)
        x = F.relu(x) 
        x = self.cv2(x)
        x = self.dp1(x)
        x = F.relu(x) 
        x = self.fc1(x)
        x = self.dp2(x)
        x = F.relu(x)
        x = self.fc2(x) 
        output = F.log_softmax(x, dim=1)
        return output 

def train(model, train_dataloader, optim, epoch, device):
    model.train()
    for b_i, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred_prob = model.forward(X)
        loss = F.nll_loss(pred_prob,y,reduction='sum').item()
        loss.backward()
        optim.step()
        if b_i%10 ==0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                epoch, b_i * len(X), len(train_dataloader.dataset),
                100. * b_i / len(train_dataloader), loss.item()))
            

def test(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        for X, y in enumerate(test_dataloader):
            X,y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss += F.nll_loss(pred_prob, y, reduction='sum').item()
            pred = pred_prob.argmax(dim=1,keepdim = True)
            success += pred.eq(y.view_as(pred)).sum().item()
    loss /= len(test_dataloader.dataset)
    print('\nTest dataset: Overall Loss: {:.4f}, Overall Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, success, len(test_dataloader.dataset),
        100. * success / len(test_dataloader.dataset)))
    
if __name__ == "__main__":
    train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,), (0.3069,))])), # train_X.mean()/256. and train_X.std()/256.
    batch_size=32, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,), (0.3069,)) 
                   ])),
    batch_size=500, shuffle=False)
device = "cpu"
model = ConvNet()
optim = optim.Adadelta(model.parameters(), lr = 0.5)
for epoch in range(1,3):
    train(model, train_dataloader, optim, epoch,device=device)
    test(model=model, test_dataloader=test_dataloader,device=device)
    
