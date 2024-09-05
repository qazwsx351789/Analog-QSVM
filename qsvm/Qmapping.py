


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
from qutip import tensor, tracedist
######################################################################################################
################################## useful operators and parameters ###################################
######################################################################################################

C6 = 5.42e-24
desire_rabi = 8*np.pi *1e6
sigx = sigmax()
sigz = sigmaz()
sigy = sigmay()
iid = qutip.qeye(2)
rr = Qobj([[0,0],[0,1]])
ee = Qobj([[1,0],[0,0]])
cnot=tensor(ee, iid)+ tensor(rr, sigx)

######################################################################################################
################################## quantum circuits and evolutioons ##################################
######################################################################################################



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

######################################################################################
################################## kernal Functions ##################################
######################################################################################


def get_q_kernel(state1 , state2 , status = "train" ):
    k_matrix = []
    for i ,s in enumerate(state1) :
        _k = []
        for j , st in enumerate(state2):
            if i==j and status == "train" :
                _k.append(1)
            elif i >= j or status == "test":
                _k.append(k_value(s,st))
            else :
                _k.append(0)
        k_matrix.append(_k)
    if status == "train" :
        for idy , km in enumerate(k_matrix) :
            for idx , k in enumerate(km) :
                if k == 0 :
                    k_matrix[idy][idx] = k_matrix[idx][idy]
    return np.array(k_matrix)

def get_q_kernel_p(state1 , state2 , gamma, atomn ,status = "train" ):
    k_matrix = []
    for i ,s in enumerate(state1) :
        _k = []
        for j , st in enumerate(state2):
            if i==j and status == "train" :
                _k.append(1)
            elif i >= j or status == "test":
                res=0
                for n in range(atomn):
                    res+=(tracedist(s.ptrace(n),st.ptrace(n)))**2
                _k.append(np.exp(-1*gamma*(res)))
            else :
                _k.append(0)
        k_matrix.append(_k)
    if status == "train" :
        for idy , km in enumerate(k_matrix) :
            for idx , k in enumerate(km) :
                if k == 0 :
                    k_matrix[idy][idx] = k_matrix[idx][idy]

    return np.array(k_matrix)

# kernel transformation, visualization
def diagnal(target , diag):
  for k in range(0,len(target)) :
    target[k][k] = diag
  return target
def rescale(target):
  _min = np.min(target)
  _max = np.max(target)
  delta  = _max - _min
  for i in range(0,len(target)) :
    for j in range(0,len(target[0])) :
      target[i][j] = (target[i][j] - _min) / delta
  return target
def show_kmatrix(test=[], train = [],name = ""):
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  # if test != [] :
  a1=axs[0].imshow(np.asmatrix(test),
                interpolation='nearest', origin='upper', cmap='Blues')
  plt.colorbar(a1)
  axs[0].set_title("testing kernel matrix")
  # if train != [] :
  a2=axs[1].imshow(np.asmatrix(train),
                interpolation='nearest', origin='upper', cmap='Blues')
  plt.colorbar(a2)
  axs[1].set_title("training kernel matrix")
  if name == "" :
    plt.show()
  else :
    plt.savefig(name)