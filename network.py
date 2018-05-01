import numpy as np
import  random

class Network(object):

    def __init__(self,size):
        self.num_layers=len(size)
        self.sizes=size
        self.weights=[np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
        self.biases=[np.random.randn(x,1) for x in size[1:]]

    def feedward(self,x):
        for w,b in zip(self.weights,self.biases):
            x=sigmoid(np.dot(w,x)+b)
        return  x

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data: n_test=len(test_data)
        n=len(training_data)
        test_accuracy=[]
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(i,self.evaluate(test_data),n_test))
                test_accuracy.append(self.evaluate(test_data)/n_test)
            else:
                print("Epoch {0} complete".format(i))
        return test_accuracy

    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights=[w-eta/len(mini_batch)*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-eta/len(mini_batch)*nb for b,nb in zip(self.biases,nabla_b)]


    def backprop(self,x,y):
        #feedward
        activation=x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        #backward
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1] =delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z=zs[-l]
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime(z)
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)



    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedward(x)),y) for (x,y) in test_data]
        return np.sum([int(x==y) for (x,y) in test_results])

    def cost_derivative(self,output_activations,y):
        return (output_activations-y)

def sigmoid(z):
    # print(z)
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
