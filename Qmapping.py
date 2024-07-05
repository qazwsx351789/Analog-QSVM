


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cmath
from numpy import random
from numpy.random import normal
from qutip import expect , Qobj
import qutip
import math
from qutip import Qobj,basis
from qutip import sigmax , sigmaz , sigmay
from qutip import tensor


######################################################################################
################################## Useful Functions ##################################
######################################################################################

sigx = sigmax()
sigz = sigmaz()
sigy = sigmay()
iid = qutip.qeye(2)
rr = Qobj([[0,0],[0,1]])
ee = Qobj([[1,0],[0,0]])
cnot=tensor(ee, iid)+ tensor(rr, sigx)

def Rz(theta):
    return Qobj([[1,0],[0,cmath.exp(1j*theta)]])

def gst(d):
  b = [basis(2,0) for i in range(d)]
  return tensor(b)
    
def form_op(tg,operator , d) :
    r = iid
    if 0 in tg :
        r = operator
    for i in range(1,d):
        if i in tg :
            r = tensor(r , operator)
        else :
            r = tensor(r , iid)
    return r
def dynamics(d,error=None):
    # if error:
    #     er=normal(loc=1.0, scale=error[1], size=1)
    #     h = form_op([0] , sigx , d)
    #     for i in range(1,d) :
    #         h += form_op([i],sigx ,d)
        
    # else:
    
    h = form_op([0] , sigx , d)
    for i in range(1,d) :
        h += form_op([i],sigx ,d)
    return h

def Hadama(d):
  h = 1/(2)**0.5*form_op([0] , Qobj([[1,1],[1,-1]]) , d)
  for i in range(1,d) :
    h = 1/(2)**0.5*form_op([i],Qobj([[1,1],[1,-1]]) ,d) *h
  return h

def EncodingP(d,data,op):
  if op=="z":
    h =  ee * cmath.exp(-1j*data[0]) + rr * cmath.exp(1j*data[0])
    for i in range(1,d) :
      h = tensor(h, ee * cmath.exp(-1j*data[i]) + rr * cmath.exp(1j*data[i]))
    return h
  elif op=="x":
    h = iid * math.cos(data[0]) - sigx * 1j*math.sin(data[0])
    for i in range(1,d) :
      h = tensor(h,iid * math.cos(data[i]) - sigx *1j* math.sin(data[i]))
    return h


def Entangle(config , d, operator_list, error=None ):
    e0 = 0
    for idx ,x in enumerate(config) :
        for idy ,y in enumerate(x[idx::],start=idx) :
            if idx != idy:
                h +=  y * operator_list[idx][idy]
            else: 
                if error:
                    e0=normal(0.0, error[0])
                try :
                    h += (y+e0) * operator_list[idx][idy]
                except :
                    h  = (y+e0) * operator_list[idx][idy]
                
    return h

def CnotGate(d):
  d-=1
  h = form_op([0] , cnot , d)
  for i in range(1,d) :
    h = form_op([i],cnot ,d) * h
  return h

def ZZGate(config,d):
  d-=1
  h = form_op([0] , cnot*tensor(iid,Rz(config[0]))*cnot , d)
  for i in range(1,d) :
    h = form_op([i],cnot*tensor(iid,Rz(config[i]))*cnot ,d) * h
  return h

def HMap(config , d, Ruby,operator_list, error=None) :
  return Ruby*dynamics(d) + Entangle(config , d,operator_list ,error)

def evolution(H,t) :
  return (-1j * H * t).expm()
    
def k_value(left , right) :
  # print(left.dag() * right)
  kk = (left.dag() * right)
  return (kk * kk.conjugate()).real

CXH = tensor((iid - sigz) , (iid - sigx))
(-np.pi * 0.25 *  1j *CXH).expm()
def noisy_cnot(d):
    AAA=[]
    d-=1
    for i in range(d) :
        aa=(CXH * normal(np.pi * 0.25 , 0.065) * -1j).expm()
        h = form_op([i] , aa , d)
        AAA.append(h)
    return AAA
    
# (-np.pi * 0.25 *  1j *CXH).expm()

def EvCnot(AAA,state):
    for i in AAA:
        state = i*state
    return state
    
def get_config(pos):
    config = np.zeros([len(pos),len(pos)])
    for idx , r in enumerate(pos) :
        for idy , _r in enumerate(pos) :
            if idx != idy :
                v = C6/((r - _r)**6)
                # since we set the value of V/U
                v = v /desire_rabi 
                config[idx][idy] = v
    return config
# chain
def noisy_pos(r, error):
    pos_x = []
    for _r in r :
        if error:
            pos_x.append(_r + normal(0,error[2]* 1e-6)) 
        else:
            pos_x.append(_r) 
    return pos_x
def evolve(H,state,t) :
    # rs = qutip.sesolve(H ,state,[0,t] )
    rs = qutip.sesolve(H ,state,np.linspace(0,t,50) )
    return rs.states[-1]



def Qmap(pos , d,t,Data ,Ruby,op,operator_list, tier,mode="quera",error=None):
    rs = []
    if mode == "ZZ":
        d+=1
    right_gst = gst(d)
    if error:
        if mode == "quera":
            dy=Ruby*dynamics(d)
            
            if tier ==0:
                for da in Data:
                    pos_n = noisy_pos (pos,error = error)  #error quera
                    config = get_config(pos_n)
                    e1=normal(loc=1.0, scale=error[1])
                    h = e1*dy +Entangle(config , d,operator_list, error )
                    # ev=evolution(h,t)
                    # state= ev * right_gst
                    state=evolve(h,state,t)
                    
                    EP=EncodingP(d,da,op)
                    state= EP * state
                    rs.append(state)
            else:
                rs = [[] for _ in range (tier)]
                for da in Data:
                    EP=EncodingP(d,da,op)
                    state= EP * right_gst
                    for i in range(tier):
                        pos_n = noisy_pos (pos,error = error)  #error quera
                        config = get_config(pos_n)
                        e1=normal(loc=1.0, scale=error[1])
                        h = e1*dy +Entangle(config , d,operator_list, error)
                        # ev=evolution(h,t)
                        # state= ev * state
                        state=evolve(h,state,t)
                        
                        state= EP * state
                        rs[i].append(state)
        elif mode == "cnot":

            if tier ==0:
                for da in Data:
                    EP=EncodingP(d,da,op)
                    Noise = noisy_cnot(lll)
                    state= EvCnot(Noise,right_gst)
                    state= EP * state
                    rs.append(state)
            else:
                rs = [[] for _ in range (tier)]
                for da in Data:
                    EP=EncodingP(d,da,op)
                    state= EP * right_gst
                    for i in range(tier):
                        Noise = noisy_cnot(d)
                        state= EvCnot(Noise,state)
                        state= EP * state
                        rs[i].append(state) 
            
            
    else:
        config = get_config(pos)
        if mode == "quera":
            h = HMap(config ,d ,Ruby,operator_list)
            ev=evolution(h,t)
        elif mode == "cnot":
            ev = CnotGate(d)
        
        if mode == "ZZ":
            Ha=Hadama(d)
            for da in Data:
              EP=ZZGate(da,d)*Ha
              state= EP * right_gst
              for i in range(tier):
                state= EP * state
              rs.append(state)
        else:
            if tier ==0:
                for da in Data:
                  EP=EncodingP(d,da,op)
                  state= EP * right_gst
                  rs.append(state)
            else:
                rs = [[] for _ in range (tier)]
                for da in Data:
                  EP=EncodingP(d,da,op)
                  state= EP * right_gst
                  for i in range(tier):
                    state= ev * state
                    state= EP * state
                    rs[i].append(state)
    return rs