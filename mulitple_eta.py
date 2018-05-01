import json
import random
import sys

import mnist_loader
import network2

import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATES=[0.025,0.25,2.5]
COLORS=['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS=30

def main():
    run_networks()
    make_plot()

def run_networks():
    random.seed(12345678)
    np.random.seed(12345678)
    training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
    training_data=list(training_data)
    validation_data=list(validation_data)
    test_data=list(test_data)
    results=[]
    for eta in LEARNING_RATES:
        print("\nTrain a network with eta=",eta)
        net=network2.Network([784,30,10],cost=network2.CrossEntropyCost())
        results.append(net.SGD(training_data,30,10,eta,lmbda=5.0,
                               evaluation_data=validation_data,
                               monitor_training_cost=True))
        f=open("multiple_eta.json","w")
        json.dump(results,f)
        f.close()

def make_plot():
    f=open("multiple_eta.json","r")
    results=json.load(f)
    f.close()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for eta,result,color in zip(LEARNING_RATES,results,COLORS):
        _,_,training_cost,_=result
        ax.plot(np.arange(NUM_EPOCHS),training_cost,"o-",
                label="$\eta$="+str(eta),color=color)
        ax.set_xlim([0,NUM_EPOCHS])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cost")
        plt.legend(loc='upper right')
        plt.show()

if __name__=="__main__":
    main()