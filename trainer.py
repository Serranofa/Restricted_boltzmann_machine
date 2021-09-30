#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy 


# In[3]:


class trainer:
    def __init__(self,nqs,hm,qm,MC,grad,nro_iterations,optimizador):
            self.nqs=nqs
            self.hm=hm
            self.qm=qm
            self.MC=MC
            self.grad=grad
            self.nro_iterations=nro_iterations
            self.optimizador=optimizador
    def train(self,nrosamples):
        #La funcion entrena a la red
        self.grad.setup()
        
        for i in range (0,self.nro_iterations):
            
            self.MC.runMC(nrosamples,i)
            
            if self.optimizador=="simple":
                shift=self.grad.parametershift(self.qm.getGradient()) #doy como parametro el gradiente de energia, y calculo DGS
                self.qm.shiftparameters(shift) #actualizo los pesos y bias
                
            elif self.optimizador=="adam":
                shift=self.grad.adam(self.qm.getGradient(),i+1) #doy como parametro el gradiente de energia, y calculo adam
                self.qm.shiftparameters(shift) # actualizo los pesos y bias
            else:
                exit
                    #self.MC.runMC(int(1e6),1)
        print(self.MC.contador)


# In[ ]:




