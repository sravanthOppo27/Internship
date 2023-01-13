import numpy as np
import math
from keras.datasets import mnist
import matplotlib.pyplot as plt
data = mnist.load_data()

(x_train,y_train),(x_test,y_test)=data

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

# normalising the given data for normalising 

x_train/=255
x_test/=255

y_train_modified = np.zeros((y_train.shape[0],10),dtype=float)

val = 0

# accuracy rate is 91.47 with 1e-4 learning rate .

learning_rate =  1e-4 

# learning_rate =  0.00020346836901064417

num_epochs = 5
  
# one hot conversion of output 

for y in y_train :
    y_train_modified[val][y]=1
    val+=1

# This function caluclates softmax derivative for the given input 

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

# This function caluclates convolution between two matrices a , b

def convolution_operation(a,b):
    w_1 = a.shape[0]
    h_1 = a.shape[1]
    w_2 = b.shape[1]
    h_2 = b.shape[2]
    out_w = w_1 - w_2 + 1
    out_h = h_1 - h_2 + 1

    output = np.zeros((b.shape[0],out_w,out_h))

    for k in range(b.shape[0]):
        for i in range(out_w):
            for j in range(out_h):
                w_start = i
                w_end = w_start + w_2
                h_start = j
                h_end = h_start + h_2
                output[k][i][j]=np.sum(np.dot(a[w_start:w_end,h_start:h_end],b[k]))

    return output

# Class for convolution layer
    
class conv_layer :

    def __init__(self,input_data_set,pad:int,stride:int,activation_funct_name:str) -> None:
        self.input_data_set = input_data_set
        self.pad = pad 
        self.stride = stride 
        self.activation_funct_name = activation_funct_name

    def filter_intialising(self,num_filter:int,w_filter:int,h_filter:int):
        self.num_filter = num_filter
        self.w_filter = w_filter
        self.h_filter = h_filter
        self.filter = np.random.normal(loc=0.0,scale=1e-4,size=(num_filter,w_filter,h_filter))

    def bias_intialising (self, input_image):
        w_inp,h_inp = input_image.shape
        num_filters,w_filter,h_filter = self.filter.shape
        bias_w = (w_inp-w_filter+(2*self.pad))//(self.stride)+1
        bias_h = (h_inp-h_filter+(2*self.pad))//(self.stride)+1
        out_w = bias_w
        out_h = bias_h
        self.bias_w = bias_w
        self.bias_h = bias_h
        self.out1_w = out_w
        self.out1_h = out_h
        bias = np.random.normal(loc=0.0,scale=1e-4,size=((self.num_filter,bias_w,bias_h)))
        self.bias = bias

    def feed_forward(self,input_image):
        self.inp1 = input_image
        w_inp,h_inp = input_image.shape
        num_filters,w_filter,h_filter = self.filter.shape
        out = np.zeros((self.num_filter,self.out1_w,self.out1_h),dtype=float)
        z1 = out

        for i in range(out.shape[0]):
            for j in range(self.out1_w):
                for k in range (self.out1_h):
                    w_start = (j*self.stride)
                    w_end = w_start+w_filter
                    h_start = (k*self.stride)
                    h_end = h_start + h_filter
                    out [i][j][k]= np.sum(np.dot(input_image[w_start:w_end,h_start:h_end],self.filter[i]))
                    z1[i][j][k]=out[i][j][k]
                    if (self.activation_funct_name=="ReLU"):
                        out[i][j][k]=max(0,out[i][j][k])

                    elif(self.activation_funct_name=="tanh"):
                        out[i][j][k]=math.tanh(out[i][j][k])

        self.out1 = out
        self.z1 = z1

        return out

    def feed_backward(self,d_loss_out1):

        d_loss_z1 = np.zeros_like(self.z1)

        for i in range (self.out1.shape[0]):
            if (self.activation_funct_name=="ReLU"):
                for j in range (self.out1.shape[1]):
                    for k in range (self.out1.shape[2]):
                            if (self.out1[i][j][k]>0):
                                d_loss_z1[i][j][k]=d_loss_out1[i][j][k]

            elif(self.activation_funct_name=="tanh"):

                d_loss_z1[i]=np.multiply(1-np.square(self.out1),d_loss_out1[i])

        self.d_loss_z1 = d_loss_z1

        self.d_loss_b1 = self.d_loss_z1

        self.d_loss_kernel = convolution_operation(self.inp1,self.d_loss_z1)

        return 

    def optimization(self):

        self.filter-= (learning_rate)*(self.d_loss_kernel)

        self.bias-=(learning_rate)*(self.d_loss_b1)

# Class for flatten layer

class flatten_layer:

    def feed_forward(self,input):

        self.input_shape = input.shape

        input = input.reshape((input.shape[0]*input.shape[1]*input.shape[2],1))

        return input

    def optimization(self):

        return 

    def feed_backward(self,input_flatten):

        input_flatten = input_flatten.reshape(self.input_shape)

        return input_flatten

# Class for dense layer

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

     

layer1 = conv_layer(x_train,0,1,"ReLU")

layer1.filter_intialising(1,3,3)

layer1.bias_intialising(x_train[0].reshape((28,28)))

layer2 = flatten_layer()

layer3 = dense_layer(10,"soft max")

layer3.intializing_dense_layer(676)

layers = [layer1,layer2,layer3]

CNN_Model = model(layers)

epochs = [1,2,3,4,5]

epoch_loses = []

epoch_accuracy=[]

# training the train data

for num in range(num_epochs):

    pos = 0

    sum=0

    count = 0

    for inp1 in x_train :

        inp1 = inp1.reshape((28,28))

        out2=CNN_Model.feed_forward(inp1)

        y_actual = y_train_modified[pos].reshape((10,1))

        if ((pos+1)%100==0):

            # print("loss for image ",pos+1," is :", np.sum(np.square(out2-y_actual))/(out2.shape[0]))

            print("loss of image " ,pos +1 , " is : ",-np.log(out2[y_train[pos]]))

        if (np.argmax(out2)==y_train[pos]):
            count+=1

        sum+=-np.log(out2[y_train[pos]])

        dx = np.zeros_like(out2)

        dx[y_train[pos]] = -1/out2[y_train[pos]]

        # d_l_out2 = out2 - y_actual

        d_l_out2 = dx

        CNN_Model.feed_backward(d_l_out2)

        CNN_Model.optimization()

        pos+=1

        # if (pos == 5000):
        #     break

    print("loss for epoch ",num+1,"is :",sum/60000)

    print("accuracy for epoch",num+1,"is :",count/60000)

    epoch_loses.append(sum/60000)

    epoch_accuracy.append(count/60000)

# testing the test data 

y_pred = np.zeros_like(y_test)

itr=0

for inp1 in x_test:

    inp1=inp1.reshape((28,28))

    out = CNN_Model.feed_forward(inp1)

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