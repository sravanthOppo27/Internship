import numpy as np
import math
from keras.datasets import mnist
import matplotlib.pyplot as plt
data = mnist.load_data()

(x_train,y_train),(x_test,y_test)=data
x_train = x_train.reshape((x_train.shape[0], 784)).astype('float64')
x_test = x_test.reshape((x_test.shape[0], 784)).astype('float64')
x_train/=255
x_test/=255

y_train_modified = np.zeros((y_train.shape[0],10),dtype=float)

val = 0

learning_rate = 1e-2

num_epochs = 5

for y in y_train :
    y_train_modified[val][y]=1
    val+=1

def softmax_derivate(mat):

    res = np.zeros((mat.shape[0],mat.shape[0]))

    res = res.astype(np.float64)

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):

            if (i==j):
                res [i][j]= mat[i][0]*(1-mat[i][0])
            else :
                res[i][j] = -mat[i][0]*mat[j][0]
    
    return res

class dense_layer :

    def __init__(self,num_nodes:int,activation_funct_name:str) -> None:
        self.num_nodes = num_nodes
        self.activation_funct_name = activation_funct_name

    def intializing_dense_layer(self,input_size:int):
        self.weights = np.random.normal(loc=0.0, scale=1e-4,size=(self.num_nodes,input_size))
        self.bias = np.random.normal(loc=0.0,scale=1e-4,size=(self.num_nodes,1))

    def feed_forward(self,input):

        self.inp2 = input

        z = np.matmul(self.weights,input)+self.bias

        if (self.activation_funct_name=="soft max"):

            val = 0

            for i in range (z.shape[0]):

                val+=math.exp(z[i][0])

            for i in range(z.shape[0]):

                z[i][0]= math.exp(z[i][0])/val

            self.out2 = z

            return z

        elif(self.activation_funct_name=="tanh"):

            for i in range(z.shape[0]):

                z[i][0]=math.tanh(z[i][0])

            self.out2 = z

            return z

    def feed_backward(self,d_l_out2):

        if (self.activation_funct_name=="soft max"):

            d_out2_z2 = softmax_derivate(self.out2)

            self.d_l_z2 = np.matmul(d_out2_z2,d_l_out2)

        elif(self.activation_funct_name=="tanh"):

            self.d_l_z2 = np.multiply((1-np.square(self.out2)),d_l_out2)


        self.d_l_w2 = np.matmul(self.d_l_z2,self.inp2.transpose())

        self.d_l_i2 = np.matmul(self.weights.transpose(),self.d_l_z2)

        self.d_l_b2 = self.d_l_z2

        return self.d_l_i2

    def optimization(self):

        self.weights-=(learning_rate)*(self.d_l_w2)

        self.bias-=(learning_rate)*(self.d_l_b2)

class model:

    def __init__(self,layers:list):

        self.layers = layers

    def feed_forward(self,inp):

        for i in self.layers:

            inp=i.feed_forward(inp)

        return inp

    def feed_backward(self,inp):

        tmp = self.layers.copy()

        tmp.reverse()

        for i in tmp:

            inp = i.feed_backward(inp)
    
    def optimization(self):

        for i in self.layers:

            i.optimization()


layer1 = dense_layer(50,"tanh")

layer1.intializing_dense_layer(784)

layer2 = dense_layer(10,"soft max")

layer2.intializing_dense_layer(50)

layers = [layer1,layer2]

ANN_model = model(layers)

epoch_loses = []

epoch_accuracy = []

epochs = [1,2,3,4,5]

# pos = 0

for num in range(num_epochs):

    sum  = 0

    pos = 0

    count = 0

    for inp1 in x_train :

        inp1=inp1.reshape((784,1))

        out=ANN_model.feed_forward(inp1)

        y_actual = y_train_modified[pos].reshape((10,1))

        # if ((pos+1)%1000==0):

        #     print("loss of image " ,pos +1 ,"for epoch ",num," is : ",-np.log(out[y_train[pos]]))


        # print("loss of image ", pos +1 , "is : ", np.sum(np.square(out2-y_actual))/(out2.shape[0]))

        if (np.argmax(out)==y_train[pos]):
          count+=1

        sum+=-np.log(out[y_train[pos]])

        dx = np.zeros_like(out)

        dx[y_train[pos]] = -1/out[y_train[pos]]

        pos+=1

        d_loss_y = out-y_actual

        ANN_model.feed_backward(d_loss_y)

        ANN_model.optimization()

    print("loss of epoch ", num+1 , "is : ",sum/60000)

    print("accuracy of epoch ",num+1 , "is :",count/60000)

    epoch_loses.append(sum/60000)

    epoch_accuracy.append(count/60000)

itr = 0

y_pred = np.zeros_like(y_test)

for inp1 in x_test:

    inp1=inp1.reshape((784,1))

    out = ANN_model.feed_forward(inp1)

    y_pred[itr]=np.argmax(out)

    itr+=1

count = 0

for i in range (y_pred.shape[0]):

    if (y_pred[i]==y_test[i]):
        count+=1

print("Accuracy is : ",(count/(y_pred.shape[0]))*100)

plt.plot(epochs,epoch_loses,color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)

plt.xlabel ('epochs')

plt.ylabel ('loses over epochs')

plt.title('loses vs epochs')

plt.show()

plt.plot(epochs,epoch_accuracy,color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)

plt.xlabel ('epochs')

plt.ylabel ('accuracy over epochs')

plt.title('accuracy vs epochs')

plt.show()