#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import NQS
import hamiltonian as hm
import Probcondicional1 as cond


# In[3]:


class Quantummodel:
    def __init__(self,nqs,ham,condi):

        self.nqs=nqs
        self.ham=ham
        self.condi=condi

        self.ndim=nqs.ndim
        self.n_hidden=nqs.n_hidden
        self.n_visible=nqs.n_visible
        self.a=nqs.a
        self.w=nqs.w
        self.b=nqs.b
        self.x=condi.x
                
        self.parameters = self.n_visible + self.n_hidden + self.n_visible*self.n_hidden
        self.sig=nqs.sig

        
        self.energyGradient=np.zeros(self.parameters)
    def zzero(self):
        # La funcion inicializa variables que voy a ir sumando desde cero.
        
        self.localEnergy        = 0;
        self.localEnergySquared = 0;
        self.acceptcount        = 0;

        self.dPsi1=np.zeros(self.parameters)
        self.energydPsi=np.zeros(self.parameters)
        
    def setupsampling(self,x):
        #The function sets up the model for a Monte Carlo simulation.
        
        self.Qsigm=self.nqs.sigmoid(x)
        dersigm=self.nqs.mdersigmoidQ()
        
        self.psi=self.nqs.psi(x)
        self.locenergy=self.ham.LocalEnergy(self.nqs)
        
        self.lap=self.nqs.laplacianalfa(x)
        
        return self.psi,self.locenergy,self.lap
    
    def acumulador(self,x):

            self.lap=self.nqs.laplacianalfa(x)
            self.locenergy=self.ham.LocalEnergy(self.nqs)
            
            self.localEnergy +=self.locenergy
            self.localEnergySquared +=self.locenergy*self.locenergy
            
            self.dPsi1+=self.lap
            self.energydPsi+=self.lap*self.locenergy

            return self.localEnergy,self.localEnergySquared,self.dPsi1,self.energydPsi

    
    def valormedio(self,nrosamples):
            self.localEnergy =self.localEnergy/nrosamples
            self.localEnergySquared =self.localEnergySquared/nrosamples
            #self.acceptcount =self.acceptcount/nrosamples
            self.dPsi1=self.dPsi1/nrosamples
            self.energydPsi=self.energydPsi/nrosamples
            
            self.var=(self.localEnergySquared-self.localEnergy*self.localEnergy)/nrosamples
            
            self.locengradient=2*(self.energydPsi-self.localEnergy*self.dPsi1)
            self.locengradientnorm=np.sqrt(np.dot(self.locengradient,self.locengradient))
                                           
    def oneBD(self,x,rmin,rmax,binwidth,onebd,ratio):
        for p in range(0,self.n_visible,self.ndim):
            r=0
            for d in range(0,self.ndim):
                r+=x[p+d]*x[p+d]
            r=np.sqrt(r)
            if (rmin<=r and r<rmax):
                binindex=int(np.floor((r-rmin)/binwidth))
                onebd[binindex]+=1
                ratio[binindex]=r
    
    def shiftparameters(self,shift):
        #The function updates the network parameters by adding a given shift.
        #It is used by the gradient descent method.
        for i in range(0,self.n_visible):
            self.a[i]=self.a[i]+shift[i]
        for j in range(0,self.n_hidden):
            self.b[j]=self.b[j]+shift[self.n_visible+j]
        k=self.n_visible+self.n_hidden
        for i in range(0,self.n_visible):
            for j in range(0,self.n_hidden):
                self.w[i,j]=self.w[i,j]+shift[k]
                k=k+1
    def newparameters(self,best):
            for i in range(0,self.n_visible):
                self.a[i]=best[i]
            for j in range(0,self.n_hidden):
                self.b[j]=best[j+self.n_visible]
            k=self.n_visible+self.n_hidden
            for i in range(0,self.n_visible):
                for j in range(0,self.n_hidden):
                    self.w[i,j]=best[k]
                    k=k+1
    def changes(self,best):
        return best
    def getGradient(self):
        return self.locengradient
    def getGradientnorm(self):
        return self.locengradientnorm


# In[ ]:




