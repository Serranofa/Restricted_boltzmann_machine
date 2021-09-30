#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import NQS


# In[10]:


class hamiltoniano:
    
    def __init__(self,omega,includeinteraction):

        self.includeinteraction=includeinteraction
        self.omega    =omega

    def OscPotential(self,x): 
    #La funcion calcula la energia potencial del oscilador armonico del sistema dada una configuracion de las particulas del sistema
   
        self.harmonicosc=np.dot(self.omega*self.omega*x,x)
        return self.harmonicosc
    
 
    def LocalEnergy(self,nqs): 
    #La funcion computa la energia del sistema descripta por la funcion de onda del objeto NQS, y de acuerdo al tipo de hamiltoniano
    #que elija
    
        harmonicoscillator =self.OscPotential(nqs.x)#self.harmonicosc
        kinetic            =nqs.Laplacian(nqs.x)
        
        localEnergy        = 0.5*(kinetic + harmonicoscillator);
        
        if (self.includeinteraction=="osc_armonico"):
            localEnergy=localEnergy
        elif (self.includeinteraction=="coulomb"):
            localEnergy          += np.sum(nqs.Inversedistance(nqs.x))
        elif (self.includeinteraction=="calogero"):
             localEnergy          +=2.1*nqs.calogero(nqs.x)
        else:
        	print("error")
        	exit      
        return localEnergy



