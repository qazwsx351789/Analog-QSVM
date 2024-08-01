import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

class Pca:
    def __init__(self,PCA_n=10,StandardizeOrNot=True):
        self.PCA_n=PCA_n
        self.StandardizeOrNot=StandardizeOrNot
        self.Fitted=False
    def fitting(self,traindata):
        if self.Fitted:
            print("It has been fitted!! The previous fitting was recovered.")
        if self.StandardizeOrNot == True:
            traindata = scaler.fit_transform(traindata)
        cov_train = np.cov(traindata.T)
        eig_vals, eig_vecs = LA.eig(cov_train)
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices]
        eig_vecs = eig_vecs[:, sort_indices]
        self.eig_vec = eig_vecs[:, :self.PCA_n]
        self.Fitted = True
    def transform(self,data):
        if self.StandardizeOrNot == True:
            data = scaler.fit_transform(data)   
        data = np.real(np.dot(data, eig_vec))
        data=np.array(data)
        return data
    def Renormalize_Each_Feature(self,data,Norm=np.pi/2):
        f_scale =[]
        f_min =[]
        
        for i in range(0 ,len(data[0])) :
          tg = [x[i] for x in data]
          f_scale.append(np.max(tg) - np.min(tg))
          f_min.append(np.min(tg))
        trs = []
        for x in data :
          rs = []
          for i , scale , min in zip(x ,f_scale , f_min) :
            rs.append((i-min)/scale*Norm)
          trs.append(rs)
        return=np.array(data)


class QSVM(self) :
    def __init__(self,method='analog+digital',task='svc'):
        self.method = method
        self.config = {}
        self.train_set = []
        if task == 'svr' :
            self.svm = SVR(method =  'precomputed')
        else:
            self.svm = SVC(method = 'precomputed')
        if self.config == {} :
            self.default_phys_sys()
    def default_phys_sys(self):
        self.config['rabi'] = 1
        self.config['detuning'] = 1
        self.config['atomn'] = 10
        self.config['pos'] = [i* 7.6*1e-6 for i in atomn]  
    def get_kernel(self, lst1 , lst2):
        if self.method == 'rbf' :
            kermnel = fsadf
            pass #sklearn
        elif self.method == 'analog'  :
            pass #analog
        elif self.method == 'digital' :
            pass #digital
        elif self.method == 'analog+digital' :
            pass #andlog+digital
        return kernel
    def fit(self,x_train , y_train) :
        self.train_set = x_train
        kernel = get_kernel(x_train , x_train)
        self.svm.fit(kernel , y_train)
    def test(self, x_test) :
        kernel = get_kernel(x_train , x_test)
        return self.svm.predict()
        
        