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
from qsvm.Qmapping import get_config
from qsvm.Qmapping import evolution
from qsvm.Qmapping import CnotGate
from qsvm.Qmapping import Hadama
from qsvm.Qmapping import ZZGate
from qsvm.Qmapping import gst
from qsvm.Qmapping import EncodingP
from qsvm.Qmapping import get_q_kernel
from qsvm.Qmapping import get_q_kernel_p
from qsvm.Qmapping import form_op
from qsvm.Qmapping import dynamics,EvCnot,evolve,HMap
from qsvm.Qmapping import noisy_pos,noisy_cnot
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


class Pca:
    def __init__(self,PCA_n=10,StandardizeOrNot=True,sklearnPCA=True,**kwargs):
        self.PCA_n=PCA_n
        self.StandardizeOrNot=StandardizeOrNot
        self.Fitted=False
        self.sklearnPCA=sklearnPCA
        if self.sklearnPCA==True:
            self.pca = PCA(n_components=PCA_n,**kwargs)
        self.scaler = StandardScaler()
    def fitting(self,traindata):
        if self.Fitted:
            print("It has been fitted!! The previous fitting was recovered.")
        if self.StandardizeOrNot == True:
            traindata = self.scaler.fit_transform(traindata)
        if self.sklearnPCA==True:
            self.pca.fit(traindata)

        else:
            cov_train = np.cov(traindata.T)
            eig_vals, eig_vecs = LA.eig(cov_train)
            sort_indices = np.argsort(eig_vals)[::-1]
            eig_vals = eig_vals[sort_indices]
            eig_vecs = eig_vecs[:, sort_indices]
            self.eig_vec = eig_vecs[:, :self.PCA_n]
            self.Fitted = True
            
    def transform(self,data):
        if self.StandardizeOrNot == True:
            data = self.scaler.fit_transform(data)
        if self.sklearnPCA==True:
            return self.pca.transform(data)
        data = np.real(np.dot(data, eig_vec))
        data=np.array(data)
        return data
        
    def Renormalize_Each_Feature(self,data,Norm=np.pi/2,status=""):

        if status == "train":
            self.f_scale =[]
            self.f_min =[]
            
            for i in range(0 ,len(data[0])) :
              tg = [x[i] for x in data]
              self.f_scale.append(np.max(tg) - np.min(tg))
              self.f_min.append(np.min(tg))
        trs = []
        for x in data :
            rs = []
            for i , scale , min in zip(x ,self.f_scale , self.f_min) :
                rs.append((i-min)/scale*Norm)
            trs.append(rs)
        return np.array(trs)


class QSVM :

    
    def __init__(self,task='svr'):
        self.task = task
        self.traindata=None
        self.trainedOrNot=False
        self.method = 'hybrid'
        self.config = {}
        self.train_set = []
        self.train_kernel=None
        self.C=1
        

    
    def default_phys_sys(self,atomn=10,aR0=1.2,detuning=0,rabi=1,t=np.pi):
        self.config['rabi'] = rabi
        self.config['detuning'] = detuning
        self.config['a/R0']=aR0
        self.config['atomn'] = atomn
        self.config['time']=t
        self.config['pos'] = [i* self.config['a/R0']*R0 for i in range(self.config['atomn'])]

    
    def get_kernel(self, data ,status='train',tier=1,method="hybrid", op="x", project=False,Error=[]):
        global fixterm
        self.Project=project
        self.tier=tier
        self.method=method
        self.Error=Error
        self.ErrorOrNot =False if self.Error==[] else True 
        
        if self.config['atomn'] != len(data[0]):
            print("warning!! The atom number is unconsistent with the number of the features ")
        rs = []
            
        if status=='training':
            if self.trainedOrNot==True:
                print("Error!!!This model has been train.")
                return 
            self.trainedOrNot=True

        # form operators
        self.operator_list = [[form_op([idx , idy] ,rr ,self.config['atomn']) for idy in range(self.config['atomn'])] for idx in range(self.config['atomn'])]
        rr_list = [form_op([idy] ,rr ,self.config['atomn']) for idy in range(self.config['atomn'])]
        
        # differnet setup
        right_gst = gst(self.config['atomn'])
        if not self.ErrorOrNot:
            config = get_config(self.config['pos'])
            
            if self.method == 'hybrid':
                h = HMap(config ,self.config['atomn'] ,self.config['rabi'],self.operator_list)
                ev=evolution(h,self.config['time'])
                
            elif self.method == "digital":
                ev = CnotGate(self.config['atomn'])
            elif self.method == "analog":
                fixterm = HMap(config ,self.config['atomn'] ,self.config['rabi'],self.operator_list,method=self.method)
        else:
            if self.method == 'hybrid' or self.method == "analog":
                dy=self.config['rabi']*dynamics(self.config['atomn'])
            
        # build states
        for da in data:
            state=right_gst
            if self.method != 'analog':     
                EP=EncodingP(self.config['atomn'],da,op)
                state= EP * right_gst
            for i in range(self.tier):
                if self.ErrorOrNot:
                    if self.method != "digital":
                        pos_n = noisy_pos (self.config['pos'],error = self.Error)  #error quera
                        config = get_config(pos_n)
                    if self.method == 'hybrid':
                        e1=normal(loc=1.0, scale=self.Error[1])
                        h = e1*dy +Entangle(config, self.config['atomn'],self.operator_list, self.Error,method=self.method)
                        state=evolve(h,state,self.config['time'])
                    elif self.method == "digital":
                        Noise = noisy_cnot(self.config['atomn'])
                        state= EvCnot(Noise,state)
                    elif self.method == 'analog':
                        e1=normal(loc=1.0, scale=self.Error[1])
                        fixterm=e1*dy+Entangle(config, self.config['atomn'],self.operator_list, self.Error,method=self.method)
                        h=add_detuning(fixterm ,rr_list, da, self.Error)
                        state=evolve(h,state,self.config['time'])

                    if self.method == 'hybrid' or self.method == "digital":
                        state= EP * state
                    
                else:
                    if self.method == 'analog':
                        h=add_detuning(fixterm ,rr_list, da, self.Error)
                        state=evolve(h,state,self.config['time'])
                    else:
                        state= ev * state
                        state= EP * state
                
                
            rs.append(state)
            
        #build kernels
        if self.Project==True:
                if status=='train':
                    self.trainState=rs
                    kernel=get_q_kernel_p(rs,rs,1/(10*data.var()),self.config['atomn'],status = "train")
                    self.train_kernel=kernel
                else:
                    kernel=get_q_kernel_p(rs,self.trainState,1/(10*data.var()),self.config['atomn'],status = "test")
        
        else:
            if status=='train':
                self.trainState=rs
                kernel=get_q_kernel(rs,rs,status = "train")
                self.train_kernel=kernel
            else:
                kernel=get_q_kernel(rs,self.trainState,status = "test")
        return kernel

    
    def fit(self,kernel, y_train, **kwargs) :
        if self.task == 'svc' :
            self.svm = SVC(kernel = 'precomputed',**kwargs)
        if self.task == 'svr' :
            self.svm = SVR(kernel =  'precomputed',**kwargs)

        self.svm.fit(kernel , y_train)
    def predict(self,te_kernel) :
        return self.svm.predict(te_kernel)
        
        