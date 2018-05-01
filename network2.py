import json
import random
import sys

import numpy as np


class QuadraticCost(object):
    def fn(self,a,y):
        return 0.5*np.linalg.norm(a-y)**2

    def delta(self,z,a,y):
        return (a-y)*sigmoid(z)

class CrossEntropyCost(object):
    def fn(self,a,y):
        return -1.0*np.sum(np.nan_to_num(y*np.log(a)+(1-y)*np.log(1-a)))

    def delta(self,z,a,y):
        return a-y

class Network(object):
    def __init__(self,sizes,cost=CrossEntropyCost):
        self.sizes=sizes
        self.num_layers=len(sizes)
        self.cost=cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        self.biases=[np.random.randn(x,1) for x in self.sizes[1:]]
        self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases=[np.random.randn(x,1) for x in self.sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feed_forward(self,x):
        for b,w in zip(self.biases,self.weights):
            x=sigmoid(np.dot(w,x)+b)
        return x

    def SGD(self,training_data,epochs,mini_batch_size,eta,lmbda,
            evaluation_data=None,
            monitor_evaluation_cost=False,monitor_evaluation_accuracy=False,
            monitor_training_cost=False,monitor_training_accuracy=False):
        if evaluation_data:n_data=len(evaluation_data)
        n=len(training_data)
        evaluation_cost,evaluation_accuracy=[],[]
        training_cost,training_accuracy=[],[]
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta,lmbda,len(training_data))
            print("Epoch %s training complete"%i)
            if monitor_training_cost:
                cost=self.total_cost(training_data,lmbda)
                training_cost.append(cost)
                print("cost on training data:{}".format(cost))
            if monitor_training_accuracy:
                accuracy=self.accuracy(training_data,convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data:{}/{}".format(accuracy,n))
            if monitor_evaluation_cost:
                cost=self.total_cost(evaluation_data,lmbda,convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data:{}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy=self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data:{}/{}".format(accuracy,n_data))
            print()
        return evaluation_cost,evaluation_accuracy,training_cost,training_accuracy


    def update_mini_batch(self,mini_batch,eta,lmbda,n):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for (nb,dnb) in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for (nw,dnw) in zip(nabla_w,delta_nabla_w)]
        self.weights=[w*(1-eta*lmbda/n)-eta/len(mini_batch)*nw for (w,nw) in zip(self.weights,nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for (b,nb) in zip(self.biases,nabla_b)]


    def backprop(self,x,y):
        #feedforward
        activation=x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            activation=sigmoid(z)
            activations.append(activation)
            zs.append(z)
        #backward
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        delta1 = self.cost.delta(z=zs[-1], a=activations[-1], y=y)
        nabla_b[-1]=delta1
        nabla_w[-1]=np.dot(delta1,activations[-2].transpose())
        for l in range(2,self.num_layers):
            delta1=np.dot(self.weights[-l+1].transpose(),delta1)*sigmoid_prime(zs[-l])
            nabla_b[-l]=delta1
            nabla_w[-l]=np.dot(delta1,activations[-l-1].transpose())
        return (nabla_b,nabla_w)



    def total_cost(self,data,lmbda,convert=False):
        cost=0.0
        for x,y in data:
            a=self.feed_forward(x)
            if convert:y=vectorized_result(y)
            cost+=self.cost.fn(a,y)/len(data)
        cost+=0.5*lmbda/len(data)*sum([np.linalg.norm(w)**2 for w in self.weights])
        return cost

    def accuracy(self,data,convert=False):
        if convert:
            results=[(np.argmax(self.feed_forward(x)),np.argmax(y)) for x,y in data]
        else:
            results=[(np.argmax(self.feed_forward(x)),y) for x,y in data]
        return sum(int(x==y) for (x,y) in results)

    def save(self,filename):
        data={"sizes":self.sizes,
              "weights":[w.tolist() for w in self.weights],
              "biases":[b.tolist() for b in self.biases],
              "cost":str(self.cost.__name__)}
        f=open(filename,"w")
        json.dump(data,f)
        f.close()

def load(filename):
    f=open(filename,"r")
    data=json.load(f)
    f.close()
    cost=getattr(sys.modules[__name__],data["cost"])
    net=Network(data["size"],cost=cost)
    net.weights=[np.array(w) for w in data["weights"]]
    net.biases=[np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    e=np.zeros((10,1))
    e[j]=1
    return e


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))