import numpy as np
import time
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

class neural_network():
    def __init__(self,sizes,epochs=100,l_rate=0.001):
        self.sizes=sizes
        self.epochs=epochs
        self.l_rate=l_rate
        self.params=self.initialization()

    def relu(self,x,derivative=False):
        if derivative:
             return 1*(x>0)         
        else:
           return  x*(x>0) 

    def softmax(self,x,derivative=False):
        exp_s=np.exp(x - x.max())
        if derivative:
            return exp_s / np.sum(exp_s,axis=0)*(1-exp_s / np.sum(exp_s,axis=0))
        else:
            return exp_s / np.sum(exp_s,axis=0)


    def initialization(self):
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]
        params={
                'w1':np.random.randn(hidden_1,input_layer)*np.sqrt(1. / hidden_1),
                'w2':np.random.randn(hidden_2,hidden_1)*np.sqrt(1. / hidden_2),
                'w3':np.random.randn(output_layer,hidden_2)*np.sqrt(1. / output_layer),
                'b1':np.random.randn(),
                'b2':np.random.randn(),
                'b3':np.random.randn()
                }
        return params
    
    def forward_pass(self, x_train):
        params=self.params
        #input layer
        params['a0']=x_train
        
        #hidden_layer 1
        params['z1']=np.dot(params['w1'],params['a0'])+params['b1']
        params['a1']=self.relu(params['z1']) 

        #hidden_layer 2
        params['z2']=np.dot(params['w2'],params['a1'])+params['b2']
        params['a2']=self.relu(params['z2'])

        #output layer
        params['z3']=np.dot(params['w3'],params['a2'])+params['b3']
        params['a3']=self.softmax(params['z3'])

        return params['a3']


    def back_pass(self,y_train,output):
        params=self.params
        change_w={}
        change_b={}

        #updated w3 and b3
        error=2*(output - y_train)/output.shape[0]*self.softmax(params['z3'],derivative=True)
        change_w['w3']=np.outer(error,params['a2'])
        change_b['b3']=error

        #updated w2 and b2
        error=np.dot(params['w3'].T,error)*self.relu(params['z2'],derivative=True)
        change_w['w2']=np.outer(error,params['a1'])
        change_b['b2']=error

        #updated w1 and b1
        error=np.dot(params['w2'].T,error)*self.relu(params['z1'],derivative=True)
        change_w['w1']=np.outer(error,params['a0'])
        change_b['b1']=error
        
        return change_w,change_b

    def update_net_param(self,change_w,change_b):
        for key,value in change_w.items(): 
            self.params[key]-=self.l_rate*value

        for key,value in change_b.items(): 
            self.params[key]-=self.l_rate*value

    def compute_accuracy(self,x_val,y_val):
        predictions=[]
        for x,y in zip(x_val,y_val):
            output=self.forward_pass(x)
            pred=np.argmax(output)
            predictions.append(pred  == np.argmax(y))

        return np.mean(predictions)


    def  train(self,x_train,y_train,x_val,y_val):
        start_time=time.time()
        for i in range(self.epochs):
            for x,y in zip(x_train,y_train):
               output=self.forward_pass(x)
               change_w,change_b=self.back_pass(y,output)
               self.update_net_param(change_w,change_b)

            accuracy=self.compute_accuracy(x_val,y_val)
            print(f"Epoch : {i+1} Time spent : {time.time()-start_time} Accuracy : {accuracy*100}")

    def save(self,init=False):
        params=self.params
        if init:
            x="init"
        else :
            x="final"    
        np.save(f'w1_{x}',params['w1'])
        np.save(f'w2_{x}',params['w2'])
        np.save(f'w3_{x}',params['w3'])
        np.save(f'b1_{x}',params['b1'])
        np.save(f'b2_{x}',params['b2'])
        np.save(f'b3_{x}',params['b3'])
        


def main():
    x,y=fetch_openml('mnist_784',version=1,return_X_y=True)
    x=(x/255).astype(np.float)
    y=to_categorical(y)
    x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.15,random_state=42)
    print(x_train)
    print(x_train.shape)
    x_tr=x_train.to_numpy()
    print(x_tr)
    print(x_tr.shape)
    arr=x_tr[4,:]
    np.save('img_test',arr)
    print(arr)
    print(arr.shape)
    n_network=neural_network(sizes=[784,128,64,10])
    #n_network.save(init=True)
    n_network.train(x_train,y_train,x_val,y_val) 
    n_network.save()
   
main()
