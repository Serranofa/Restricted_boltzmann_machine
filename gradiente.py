#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import NQS
import math


# In[4]:


class gradiente:
    
    def __init__(self,learningrate,gamma,nqs):
        self.nqs=nqs
        self.n_hidden=nqs.n_hidden
        self.n_visible=nqs.n_visible
        self.eta=learningrate
        self.gamma=gamma
        self.parameter=self.n_hidden+self.n_visible+(self.n_visible*self.n_hidden)
        self.shift=np.zeros(self.parameter)
        
        self.epsilon=1e-8
        self.beta1=0.9
        self.beta2=0.99
        self.m=np.zeros(self.parameter)
        self.s=np.zeros(self.parameter)
        self.squared=np.zeros(self.parameter)
        
        
    def parametershift(self,gradient):
    #The function computes the shift with which the network 
    #parameters should be updated according to 
    #the simple gradient descent algoirthm.
    
        self.shift=self.gamma*self.prev_shift+self.eta*gradient
        self.prev_shift=self.shift
        
        return -self.shift
    
    def adam(self,gradient,iteration):
        self.m=self.beta1*self.prev_m+(1-self.beta1)*gradient
        for i in range(0,self.parameter):
            self.squared[i]=gradient[i]*gradient[i]
            
        self.s=self.beta2*self.prev_s+(1-self.beta2)*self.squared
        self.prev_m=self.m
        self.prev_s=self.s
        
        self.m=self.m/(1-math.pow(self.beta1,iteration))
        self.s=self.s/(1-math.pow(self.beta2,iteration))
        #print(prev_m[1],)
        for i in range(0,self.parameter):
            self.shift[i]=self.m[i]/(np.sqrt(self.s[i])+self.epsilon)
        return -self.eta*self.shift
    def setup(self):
        self.prev_m=np.zeros(self.parameter)
        self.prev_s=np.zeros(self.parameter)
        self.prev_shift=np.zeros(self.parameter)


# In[ ]:




