#coding: utf-8
#import numpy as np
from numpy import *

class BaseHashing(object):

    def __init__(self, **kw):
        for key, value in kw.iteritems():
            setattr(self, key, value)

class Hashing(BaseHashing):
    """

    """
    def __init__(self, b, t, **kw):
        #super(BaseHashing,self).__init__(b=b, t=t, **kw)
        #self.storage = PickleStorage()

        self.b = b
        self.t = t

    def kernelMatrix(self, X, Y=None, gma=1, kernel='rbf'):
        if Y is None:
            Y = X
        if kernel == 0:
            X = mat(X)
            Y = mat(Y)
            return X*(Y.T)
        else:
            X = mat(X)
            Y = mat(Y)
            m = X[:,0].size
            n = Y[:,0].size
            onesm = mat(ones(m))#m长行向量,X矩阵的数据个数
            onesn = mat(ones(n))
            K = sum(multiply(X,X),axis=1) * onesn #(m*1) * (1*n)
            #mat(sum(multiply(X,X),axis=1))为n*1矩阵
            #sum axis=1按行相加
            #np.multiply对应元素相乘
            K = K + onesm.T * sum(multiply(X,X),axis=1).T  #(m*1) * (1*n)
            K = K - 2*mat(X)*mat(Y).T #(m*d) * (d*n)
            if gma is None:
                scale = sum(sqrt(abs(K)))/(m**2-m)
                gma = 1/(2*scale**2)
            return exp(-gma*K)

    def center(self, K):
        m = K[:,0].size
        onesm = mat(ones((m,m)))
        return K - (onesm*K + K*onesm - sum(K)/m)/m

    def do_hashing(self, X, vector, W):
        kernelized_vector = self.kernelMatrix(X, vector)
        array = kernelized_vector * W
        array = maximum(array, 0)
        hashed_array = sign(array)
        return hashed_array

    #@property
    def creatHashTable(self, K, b, t):
        '''
        Inputs: K (kernel matrix)
                b (number of bits)
                t (number of Gaussian approximation elements)
        Outputs:H (hash table over elements of K, size p x b)
                W (weight matrix for computing hash keys over queries, size p x b)
        '''
        #initialization
        p = K[:,0].size
        W = mat(zeros((p,b)))

        eigVals,eigVects = linalg.eig(K)#eigVals is an array
        #eigValInd = eigVals.argsort()[::-1]
        #eigVals = eigVals[eigValInd]
        #eigVects = eigVects[:,eigValInd]

        boolindex = eigVals > 1e-8
        eigVals[boolindex] = sqrt(eigVals[boolindex])
        vals_k = mat(diag(eigVals))
        K_half = eigVects*vals_k*eigVects.T

        #create indices for the t random points for each hash bit
        #and form weight matrix
        for i in xrange(b):
            #generate random permutation
            perm = arange(p)
            random.shuffle(perm)#random permutation
            I_s = perm[0:t]
            e_s = zeros((p,1))
            e_s[I_s] = 1
            W[:,i] = K_half*e_s

        H = K * W
        H = maximum(H, 0)#set element(<0) = 0
        HashTable = sign(H)
        return (HashTable,W)