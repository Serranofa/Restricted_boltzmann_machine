#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import Probcondicional1 as cond
import quantumodel1 as quam


# In[5]:


class MCMethod:
    def __init__(self,qm,seed):
        self.condi=qm.condi
        self.qm=qm
        self.seed=seed
        self.contador=0
    def runMC(self,nrosamples,nro):
        onebdd=False
        
        self.qm.zzero()
        self.qm.setupsampling(self.qm.condi.x)
        
        effectiveNsamples=0
        file=open("coulumb3d1.txt","a")
        equilibration=False
        
        for sample in range(0,nrosamples):
            self.qm.condi.gibbs(self.seed)
            equilibration=False
            if sample> 0.1*nrosamples: #empiezo a calcular los valores de energia 
                equilibration=True
            if equilibration==True:
                self.qm.acumulador(self.qm.condi.x)
                #file.writelines(str(self.qm.localEnergy))
                
                effectiveNsamples+=1
        
        self.qm.valormedio(effectiveNsamples) #obtengo el valor medio de la energia, y calculo la varianza y el gradiente
        
    
        print(f"valor de E= {self.qm.localEnergy:0.9f}",self.qm.locengradientnorm,nro)
        file.write(f"{self.qm.localEnergy:0.7f}"+"  "+f"{self.qm.var:0.7e}"+"  "+f"{np.sqrt(self.qm.var):0.7e}"+"  "+f"{self.qm.locengradientnorm:0.7f}"+"  "+f"{np.abs(self.qm.localEnergy-2):0.7f}"+"  "+"  "+str(nro)+"\n")
        
        if self.qm.localEnergy<2.: #contador que te permite ver cuando estas por debajo del valor de energia deseado
            
           self.contador+=1





