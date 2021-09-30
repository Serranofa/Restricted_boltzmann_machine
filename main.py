# In[1]:


import numpy as np
import NQS
import hamiltonian as hm
import quantumodel1 as quam
import MCMethod1 as MCM
import gradiente as gradi
import Probcondicional1 as cond
import trainer
import time
#import GA1


# In[2]:
#Se pasan los parámetros que el algoritmo necesita, omega es la frecuencia del oscilador armónico, sigma es varianza de mi gaussiana incial, el tipo de interaccion "osc_armonico", "coulomb" o "calogero",la distribucion incial de los pesos y bias "normal" o "uniforme", cantidad de particulas y dimension (unidades visibles=nro de particulas+dimension) y el numero de unidades ocultas.

#model
Omega                    = 0.1      
sigma                    = 1/np.sqrt(2*Omega)
coulombinteraction       = "coulomb" 
nqsInitialization        = "normal"
nParticles               = 2
nDimensions              = 3
nHidden                  = 4
seed1                    =np.random.seed()

#Method
#Se indica la cantidad de sampleos de Gibbs que se va a realizar
numberOfSamples = int(1e5)
seed2           = np.random.seed()
contador        =0

#Trainer
#Se elige el tipo de optmizacion "adam","simple", de ahi indicamos la cantidad de veces que vamos a realizar el metodo, incluimos el valor del learningrate y gamma que esta asociado al momento (solo si elijo la opcion "simple").
minimizertype   = "adam"
nIterations     = 450;
learningrate    = 0.05;
gamma           = 0.;




# In[3]:
#llamo a las clases y ejecuto mi programa con los parametros especificados anteriormente.

nqs=NQS.NQS(nHidden,nDimensions,nParticles,sigma)
grad=gradi.gradiente(learningrate,gamma,nqs)

nqspositive=cond.NQSpositive(nqs)
ham=hm.hamiltoniano(Omega,coulombinteraction)
qm=quam.Quantummodel(nqs,ham,nqspositive)
MC=MCM.MCMethod(qm,np.random.seed())
nqs.initi(nqsInitialization,seed1)
#ga=GA1.genetic(nqs,qm)
t=trainer.trainer(nqs,ham,qm,MC,grad,nIterations,minimizertype)
tic=time.perf_counter()
t.train(numberOfSamples)
toc=time.perf_counter()
print(contador)
print(f"Duracion total= {toc - tic:0.4f} seconds")




