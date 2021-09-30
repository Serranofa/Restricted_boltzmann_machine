#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import NQS


# In[ ]:


class NQSpositive:
    def __init__(self,nqs):
        self.nqs=nqs
        
        self.n_hidden=nqs.n_hidden
        self.n_visible=nqs.n_visible
        self.a=nqs.a
        self.w=nqs.w
        self.x=nqs.x
        self.sig=nqs.sig
        self.h=nqs.h
        
        self.normal_dist=np.zeros([])

    
    def probHgivenX(self):
        #La funcion devuelve la probabilidad condicional de h_j=1, dado el valor de x
        self.Qsigm=self.nqs.sigmoidQ

        return self.Qsigm
    
    def probXgivenH(self,i):
        #La funcion devuelve la distribucion de probabilidad condicional de cada valor de x dado el valor de h
        xMean = self.nqs.a[i] + np.dot(self.nqs.w[i,:],self.h)
        return np.random.normal(xMean,self.sig)
    
    def gibbs(self,seed2):
        #La funcion samplea una nueva configuracion de posiciones de acuerdo al metodo de Gibbs
        #Da el valor de nuevas unidades ocultas de acuerdo a la funcion sigmoide comparando esta con un valor aleatorio entre 0 y 1
    
        self.Qsigm=self.nqs.sigmoidQ
        for j in range(0,self.n_hidden):
            if  self.Qsigm[j] > np.random.rand():
                  self.h[j]=1.0
            else:
                self.h[j]=0.
                 
        for i in range(0,self.n_visible):
            self.x[i]=NQSpositive.probXgivenH(self,i)
        self.Qsigm=self.nqs.sigmoid(self.x)
        dersigm=self.nqs.mdersigmoidQ()
        return self.x,self.h

