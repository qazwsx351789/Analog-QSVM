import numpy as np
import matplotlib.pyplot as plt

# Scikit Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error as mse
import time





####################################################################################
################################## data functions ##################################
####################################################################################
class progress_bar() :
    def __init__(self ,length):
        self.length = length
        self.done = "#" *20
        self.task = "-" *20
        self.clock = -1
        self.progress = -1
        self.time = time.time()
    def click(self):
        self.clock += 1
    def update(self) :
        self.click()
        p = int(20 *self.clock /self.length)
        if p != self.progress :
            print("[" ,self.done[0:p] , self.task[p:20] ,"]" , end = '\r')
            self.progress = p
    def duration(self):
        print("\ntime : "  ,(time.time() - self.time))

def shift(target):
  _min = np.min(target)
  return [t-_min for t in target]
def _filter(target , filter) :
  r = []
  rid = []
  rest = []
  for idx , t in enumerate(target) :
    if t > filter :
      r.append(t)
      rid.append(idx)
    else :
      rest.append(t)
  return r ,rid , rest
def indexing(target , index) :
  return [target[idx] for idx in index]
def _plot(x,y , filter,color = [] , name = "",start=0,end=0.1):
  px = []
  py = []
  for _x , _y in zip(x,y) :
    if abs(_y) < filter :
      px.append(_x)
      py.append(_y)
  if color == [] :
    plt.scatter( px, py)
  else :
    plt.scatter(px ,py , c =color)
  plt.plot(np.linspace(start,end) , np.linspace(start,end))
  if name != "" :
    plt.savefig(name)
  plt.show()
def _plot_p(x,y , filter,color = [] , name = ""):
  px = []
  py = []
  for _x , _y in zip(x,y) :
    if abs(_y) < filter :
      px.append(_x)
      py.append(abs(_y))
  if color == [] :
    plt.scatter( px, py)
  else :
    plt.scatter(px ,py , c =color)
  plt.plot(np.linspace(0,0.1) , np.linspace(0,0.1))
  if name != "" :
    plt.savefig(name)
  plt.show()


def tune(target ,x_train , y_train,kernel_method) :
  param_grid =[{'kernel':kernel_method,
      'epsilon':np.arange(0,1,0.001),
             'C':np.arange(0.1,1,0.1)}]
  # 初始化网格搜索的方法
  rd_search = RandomizedSearchCV(target ,param_grid,n_jobs=8)
  rd_search.fit(x_train, y_train)
  return rd_search.best_params_
def _tune(x_train,y_train,x_test , y_test ,kernel_method , tg = 'ts'):
  param_grid =[{'kernel':kernel_method,
    'epsilon':np.arange(0,0.1,0.001),
            'C':np.arange(0.1,1,0.1)}]
  best = 9999
  best_dict = {'kernel': kernel_method[0]}
  for ep in (param_grid[0]['epsilon']):
    for c in param_grid[0]['C']:
      _svr = SVR(kernel = kernel_method[0] , epsilon = ep , C=c)
      _svr.fit(x_train , y_train)
      if tg == 'ts':
        _rs = _svr.predict(x_test)
        _sc = mse(_rs , y_test)
      else :
        _rs = _svr.predict(x_train)
        _sc = mse(_rs , y_train)
      if _sc < best :
        best = _sc
        best_dict['epsilon'] = ep
        best_dict['C'] = c
  # print(param_grid[0]['C'])
  return best_dict

def pca():
    scaler = StandardScaler()
    train = scaler.fit_transform(traindata)
    test = scaler.fit_transform(testdata)
    
    #step2
    cov_train = np.cov(train.T)
    
    #step3
    eig_vals, eig_vecs = LA.eig(cov_train)
    
    #step4
    sort_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sort_indices]
    eig_vecs = eig_vecs[:, sort_indices]
    
    #step5
    k = 10 ## The number of features after PCA
    eig_vec = eig_vecs[:, :k]
    
    
    #step6
    train_pca = np.real(np.dot(train, eig_vec))
    test_pca = np.real(np.dot(test, eig_vec))
