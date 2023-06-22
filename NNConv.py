import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


epochs = 10
batch = 100
learning_rate = 0.01


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= batch , shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayer1 = nn.Conv2d( 1, 8 , 3 , 1, 1  )
        
        self.drop = nn.Dropout()
        self.pooling = nn.MaxPool2d(2,2)
        self.convlayer2  = nn.Conv2d(8, 16 , 3 , 1 , 1)
        self.convlayer3 = nn.Conv2d(16,32, 3, 1 , 1)
        self.FC = nn.Linear(32*7*7, 10)

    def forward(self , x ):
        x = self.pooling( F.relu(self.convlayer1(x)))
        x = self.drop(x)
        x = self.pooling( F.relu(self.convlayer2(x)))
        x = F.relu(self.convlayer3(x))
        x = x.view(-1 , 32 * 7 * 7 )
        x = self.FC(x)
        return x 


    
model = CNN().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

total_steps = len(train_loader) 
losses = []
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #plt.imshow(torch.squeeze(outputs[0]).cpu().detach().numpy())
        
        
        loss = loss_function(outputs, labels)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'epoch {epoch + 1 }/  { epochs} , step {i+1} / {total_steps}, loss = {loss.item()}' )
image_misclassified = []
misclassified = []
correct_labels = []
with torch.no_grad():
    n_correct = 0 
    n_samples = 0
    for images, labels in test_loader: 
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples  += labels.shape[0]
        n_correct = (predictions == labels).sum().item()

    acc = 100 * n_correct/ n_samples
    print(f'accuracy = {acc} ') 

    