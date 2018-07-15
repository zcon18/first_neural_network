
import numpy as np
import random
import tensorflow as tf
def sig(x):
    o=[]
    for i in range(len(x)):
        o.append(1.0 / (1 + np.exp(x[i])))
    return o

def sigD(x):
    o=[]
    for i in range(len(x)):
        o.append(x[i] * (1.0 - x[i]))
    return o
def randMulti(s,e,n): #this creates a random start value for the weights and biases
    o=[] #creates an output array
    for i in range(n):
        o.append(random.randint(s,e)) #for the start and end values of random it adds a new value to the end of ray
    return o
def randMulti2(s,e,n1,n2):
    o=[[]]
    t=[]
    for i in range(n1):
        for j in range(n2):
            t.append(random.randint(s,e))
        o.append(t)
        t=[]
    return o
def Lcal(a,w,b): #this caculates the activation for each node in each layer minus the sigmoid (z)
    a2=[] #activation 2 for output
    for i in range(len(a)-1):
        for j in range(len(w[i])-1):
            a2.append(a[i]*w[i][j]) #caculates a new vaule in the output array
        a2[i]+=b[i] #adds the bais for each node *notes how it is still in the outer for loop
    return a2
def Dbais(z,a,y): #Caculates the pratial dervitive for one bias for one test case
    return sigD(z)*2*(a-y)
def Dbais(z,x):
    return sigD(z)*x
def Dweight(apre,z,a,y): #Cain rule for weights
    return apre*sigD(z)*2*(a-y)
def Dweight(apre,z,x):
    return apre*sigD(z)*x
def Dact(con,w,z,a,y): #because one node will affect more than one node in the output layer you need to take thoses spacifice weights
    o=0
    for i in range(len(w)-1):
        o += w[con][i] * sigD(z[i]) * 2 * (a[i] - y)
    return o
def Dact(con,w,z,x):
    o=0
    for i in range(len(w)-1):
        o += w[con][i] * sigD(z[i]) * x
    return o
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x[[]]
        self.W1 =randMulti2(0.00,1.00,16,784) #the what ever 784*16 weights connecting each 784 nodes in the input and 16 nodes in the first hidden layer
        self.B1 =randMulti(0.00,1.00,16) #the 16 biases in the first hidden layer
        self.W2 =randMulti2(0.00,1.00,16,16) #the 16*16 weights connecting hidden layer 1 to hidden layer 2
        self.B2 =randMulti(0.00,1.00,16) #the 16 biases in the second hidden layer
        self.W3 =randMulti2(0.00,1.00,10,16) #the 16*10 weights connecting hidden layer 2 to output
        self.B3 =randMulti(0.00,1.00,10) #the 10 biases for the output layer
        self.y          = y[[]] #lets say the first in puts are 9 then 3 it would look like: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.output     = np.zeros(self.y.shape)
        self.times = 0
        self.mem=[[]] #This holds all of the activation values for every test case. Each first demention is a single test case, which the elements inside are values
        self.alpha = 0 #this is the learning rate

    def feedforward(self):
        self.Z1=Lcal(self.input[self.times], self.W1, self.B1) #16 nodes in hidden layer 1
        self.A1 =sig(self.Z1) #Applying Activation fuction to hiddlen Layer 1
        self.Z2 = Lcal(self.A1, self.W2, self.B2) #16 nodes in hidden layer 2
        self.A2 = sig(self.Z2) #Actviation fuction to hidden layer 2
        self.Z3 = Lcal(self.A2, self.W3, self.B3) #10 nodes in output
        self.output =sig(self.Z3) #Final activation fuction
        self.times+=1 #this keeps track of what element in the batch we are in gets reset later
        self.temp=[] #this is a temp varable for holding all the of the activiation values for 1 test which gets added as a new 1st demention of mem, and gets reset everytime the feedforward is called
        #for i in range(16):
            #self.temp.append(self.A1[i]) #adds hidden layer 1 activation
        #for i in range(16):
            #self.temp.append(self.A2[i]) #adds hidden layer 2 activation
        for i in range(10):
            self.temp.append(self.output[i]) #adds output layer activation
        self.mem.append(self.temp)
    def cost(self):
        self.temp2=0 #this temp vairble will hold the un-averaged sum
        for i in range(self.times): #This outer loop will run for how many times we ran feedforward (aka, batch size)
            for j in range(len(self.output)): #this outer for loop will run for only ten times to get only the output layer values
                self.temp2+=((self.mem[i][j]-self.y[i][j])*(self.mem[i][j]-self.y[i][j])) #This grabs both the output activation values, minus what we want the outputs to be, and squares it.
        return self.temp2/self.times #avarage
    def backprop(self):
        self.temp3=[[]] #This is a 2d temp list to hold all the test cases
        self.addboi=[] #This is a second temp list to hold the added case right before we avarage
        self.gradV=[] # this holds the vector
        self.holder=0 #this is used to grab the correct number in the vertor for a bias/weight
        for i in range(len(self.mem)-1):
            for j in range(len(self.mem[i])-1):
                for k in range(len(self.B3)-1):
                    self.temp3[i].append(Dbais(self.Z3[j], self.mem[i][j], self.y[i][j]))
                for k in range(len(self.W3[j])-1):
                    self.temp3[i].append(Dweight(self.A2[j], self.Z3[j], self.mem[i][j], self.y[i][j]))
                for k in range(len(self.B2)-1):
                    self.temp3[i].append(Dbais(self.Z2[k],Dact(k,self.W3,self.Z3,self.mem[i][j],self.y[i][j])))
                for k in range(len(self.W2)-1):
                    for l in range(len(self.W2[k])-1):
                        self.temp3[i].append(Dweight(self.A1[j], self.Z2[j], Dact(k, self.W3, self.Z3, self.mem[i], self.y[i][j])))
                for k in range(len(self.mem[i])-1):
                    for l in range(len(self.B1)-1):
                        self.temp3[i].append(Dbais(self.Z1[l], Dact(l, self.W2, self.Z2, Dact(k, self.W3, self.Z3, self.mem[i], self.y[i][j]))))
                for k in range(len(self.mem[i])-1):
                    for l in range(len(self.W1)-1):
                        for m in range(len(self.W1[l])-1):
                            self.temp3[i].append(Dweight(self.input[i][j], self.Z1[l], Dact(l, self.W2, self.Z2, Dact(k, self.W3, self.Z3, self.mem[i], self.y[i][j]))))
        for i in range(len(self.temp)-1):
            for j in range(len(self.temp[i])-1):
                self.addboi[i]+=self.temp[j][i]
            self.gradV.append(self.addboi[i]/len(self.temp[i])-1)

        for i in range(len(self.B3)-1):
            self.B3[i]-=self.alpha*self.gradV[self.holder]
            self.holder+=1
        for i in range(len(self.W3)-1):
            for j in range(len(self.W3[i])-1):
                self.W3[i][j]-=self.alpha*self.gradV[self.holder]
                self.holder+=1
        for i in range(len(self.B2)-1):
            self.B2[i]-=self.alpha*self.gradV[self.holder]
            self.holder+=1
        for i in range(len(self.W2)-1):
            for j in range(len(self.W2[i])-1):
                self.W2[i][j]-=self.alpha*self.gradV[self.holder]
                self.holder+=1
        for i in range(len(self.B2)-1):
            self.B1[i]-=self.alpha*self.gradV[self.holder]
            self.holder+=1
        for i in range(len(self.W1)-1):
            for j in range(len(self.W1[i])-1):
                self.W1[i][j]-=self.alpha*self.gradV[self.holder]
                self.holder+=1


if __name__ == "__main__":
    #from this line to the "end" I copied code from the pre-build function to grab images from the minst dataset
    image_height = 28
    image_width = 28

    color_channels = 1

    model_name = "mnist"

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    x = mnist.train.images
    y = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    category_names = list(map(str, range(10)))
    x = np.asarray(x)
    print(x.shape)
    print(x[0])
    #end
    nn = NeuralNetwork(x,y)
    for i in range(len(x)):
        for j in range(1500):
            nn.feedforward()
        print(nn.cost)
        nn.backprop()


