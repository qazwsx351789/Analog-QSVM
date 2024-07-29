class QSVM(self) :
    def __init__(self,method,task='svr',config={}):
        self.method = method
        self.config = config
        self.train_set = []
        self.svm = SVC(method = 'precomputed')
        if task == 'svr' :
            self.svm = SVR(method =  'precomputed')
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
        
        