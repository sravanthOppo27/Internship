#  Implementation of CNN model for classification of mnist data set 

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable

train_data = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)

test_data = datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

batch_size = 100

num_iterations = 3000

batches_per_epoch = len(train_data)/(batch_size)

num_epochs = (num_iterations)/batches_per_epoch

num_epochs=(int)(num_epochs)

train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

test_loader  = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

class CNN_Model(nn.Module):
    def __init__(self) -> None:
        super(CNN_Model,self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=0)

        self.relu1 = nn.ReLU()

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=0)

        self.relu2=nn.ReLU()

        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.dense_1 = nn.Linear(32*4*4,50)

        self.dense_2 = nn.Linear(50,10)

        self.queue = [self.cnn1,self.relu1,self.max_pool_1,self.cnn2,self.relu2,self.max_pool_2]

        # self.full_connected = nn.Linear(32*4*4,10)
    
    def forward(self,inp):

        for q in self.queue :
            inp = q(inp)

        inp = inp.view(inp.size(0),-1)

        inp = self.dense_1(inp)

        inp=self.dense_2(inp)

        return inp


model = CNN_Model()

loss = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# print(optimizer)

for epoch in range (num_epochs):

    sum = 0

    total = 0

    for i,(image,label) in enumerate(train_loader):

        image = Variable(image)

        label = Variable(label)

        optimizer.zero_grad()

        output = model(image)

        loss_value = loss(output,label)

        loss_value.backward()

        optimizer.step()

        sum+=loss_value.item()

        total +=label.size(0)

        if ((i+1)%100==0):

            print("Epoch [{}/{}] , Batch [{}/{}] , loss : ".format(epoch+1,num_epochs,i+1,batches_per_epoch),loss_value.item())

    print("Loss for epoch ",epoch+1,"is : ",sum/batches_per_epoch)

total_number = 0

count = 0

for i,(image,label) in enumerate(test_loader):

    image = Variable(image)

    label = Variable(label)

    output = model(image)

    _,predicted = torch.max(output,axis=1)

    total_number +=label.size(0)

    count+=(predicted==label).sum() 

print("Test Set accuracy is %.4f"%((count/total_number)*100))