import numpy as np
from numpy import linalg as LA
from numpy.random import normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import qutip
from qutip import expect , Qobj
from qutip import tensor, qeye
from qutip import sigmax , sigmaz , sigmay

from sklearn.svm import SVC
from sklearn.svm import SVR

import qsvm.Qmapping as Q
import qsvm.data_function as df
from qsvm.Qmapping import get_config,evolution,CnotGate,Hadama,ZZGate,gst
from qsvm.Qmapping import EncodingP,get_q_kernel,form_op,dynamics
from qsvm.Qmapping import EvCnot,evolve,HMap,noisy_pos,noisy_cnot
from qsvm.Qmapping import Entangle,add_detuning

C6 = 5.42e-24
desire_rabi = 8*np.pi *1e6
R0=(C6/desire_rabi)**(1/6)

sigx = sigmax()
sigz = sigmaz()
sigy = sigmay()
iid = qeye(2)
rr = Qobj([[0,0],[0,1]])
ee = Qobj([[1,0],[0,0]])
cnot=tensor(ee, iid)+ tensor(rr, sigx)

class QDataGenerator:

    def __init__(self,feature_n=10,data_n=1000,loc=0,scale=1.0,data=[]):
        self.data_n=data_n
        self.feature_n=feature_n
        self.loc=loc
        self.scale=scale
        self.data=data
        if data==[]:
            self.data=[normal(loc=loc,scale=scale,size=self.feature_n) for _ in range(self.data_n)]

            
    
    def encoding_data(self,op="x",encoding_method='digital'):
        self.encoding_method=encoding_method
        self.op=op
        self.encoding_states=[]
        right_gst = gst(self.feature_n)
        if self.encoding_method == 'digital':
            for da in self.data:
                state=right_gst   
                EP=EncodingP(self.feature_n,da,self.op)
                state= EP * state
                self.encoding_states.append(state)
  
    def generate_QNN(self,tier=10,QNN_method="analog",atomn=10,aR0=1,detuning=0,rabi=1,t=np.pi):
        self.config={}
        self.tier=tier
        self.QNN_method=QNN_method
        self.config['rabi'] = normal(loc=rabi,scale=1.0,size=self.tier)
        # self.config['detuning'] = normal(loc=detuning,scale=1.0,size=self.tier)
        self.config['a/R0']=aR0
        self.config['atomn'] = atomn
        self.config['time']=t
        self.config['pos'] = [i* self.config['a/R0']*R0 for i in range(self.config['atomn'])]
        self.QNN_list=None
        # form operators
        self.operator_list = [[form_op([idx , idy] ,rr ,self.config['atomn']) for idy in range(self.config['atomn'])] for idx in range(self.config['atomn'])]
        rr_list = [form_op([idy] ,rr ,self.config['atomn']) for idy in range(self.config['atomn'])]
        
        # differnet setup
        right_gst = gst(self.config['atomn'])
        config = get_config(self.config['pos'])
        if self.QNN_method == 'analog':
            h = [HMap(config ,self.config['atomn'] ,self.config['rabi'][i],self.operator_list) for i in range(self.tier)]
            self.QNN_list=[evolution(h[i],self.config['time']) for i in range(tier)]\
        
    def generate_labels(self,observable="z",atom_ind=1):
        # build states
        self.atom_ind=atom_ind
        self.final_state=[]
        self.labels_list=[]
        self.observable=observable
        for state in self.encoding_states:
            state_=state
            for tier in range(self.tier):
                state_= self.QNN_list[tier] * state_
            self.final_state.append(state_)
        if self.observable=="z":
            self.labels_list=[(sigz*state.ptrace(atom_ind)*state.ptrace(atom_ind).dag()).tr() for state in self.final_state]
        if self.observable=="x":
            self.labels_list=[(sigx*state.ptrace(atom)*state.ptrace(atom).dag()).tr() for state in self.final_state]