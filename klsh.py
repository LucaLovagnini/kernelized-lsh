#coding: utf-8
"""
DESCRIPTION
===========
This module is a Python implementation of
Kernelized Locality Sensitive Hashing,
which is a alpha version.

AUTHOR
===========
Jason Ding(ding1354@gmail.com)

LICENCE
===========
This module is free software that you can use it or modify it.
"""
import numpy as np
#import math
from hashing_bits import Hashing
from buckets import KlshBucket
from storage import PickleStorage

class KLSHBase(object):
    '''
    Base class for KLSH
    '''
    def __init__(self, **kw):
        for key, value in kw.iteritems():
            setattr(self, key, value)

    def _check_parameters(self):
        for param in ('b', 't'):
            if not hasattr(self, param):
                raise TypeError("parameter 'param' is necessary")

class KLSH(KLSHBase):
    '''
    >>>from klsh import KLSH
    >>>klsh = KLSH(
    ...     b=b,    #number of hash bits
    ...     t=t,    #number of
    ...)
    '''
    def __init__(self, b, t, **kw):
        super(KLSH,self).__init__(b=b, t=t,**kw)
        #self._check_parameters()

        self.hash = Hashing(b=b, t=t)
        self.bucket = KlshBucket()
        self.storage = PickleStorage()

    def loadDataSet(self,filename,delim=','):
        self.dataMat =  np.loadtxt(filename, delimiter=delim)

    def preprocessing(self):
        dataMat = self.dataMat
        numOfData = dataMat[:,0].size
        self.insert_matrix(dataMat, numOfData)
        self.store_Wmat()
        self.bucket.store_buckets()


    def insert_matrix(self, matrix, num):
        KerMat = self.hash.kernelMatrix(matrix)
        CenterMat = self.hash.center(KerMat)
        (HashTable,self.W) = self.hash.creatHashTable(CenterMat, self.b, self.t)
        for i in xrange(num):
            self.bucket.insert_buckets(matrix[i,:], HashTable[i,:])

    def knn(self, vector, knum, stored=False):
        if stored == False:
            hashed_array = self.hash.do_hashing(vector, self.dataMat, self.W)#vector must be the first argument
            knn_vectors = self.bucket.select_knn(knum, hashed_array)
        else:
            self.W = self.load_Wmat()
            self.bucket.load_buckets()
            hashed_array = self.hash.do_hashing(vector, self.dataMat, self.W)
            #hashed_array = self.hash.do_hashing(self.dataMat, vector, self.W)
            knn_vectors = self.bucket.select_knn(knum, hashed_array)

        return knn_vectors

    def store_Wmat(self):
        fw = open('Wmat.data','wb')
        self.storage.save(self.W, fw)
        fw.close()

    def load_Wmat(self):
        fr = open('Wmat.data')
        W = self.storage.load(fr)
        fr.close()
        return W



if __name__ == '__main__':
    klsh = KLSH(b=200, t=30)
    klsh.loadDataSet("training_set.txt")
    #klsh.preprocessing()
    vect = [1,10,1,11,1,13,1,12,1,1,9]
    vect = np.mat(vect,dtype=np.float64)
    #knn_vects = klsh.knn(vect,3)
    knn_vects = klsh.knn(vect,3,stored=True)
    #knn_vects = klsh.knn(vect,3,stored=False)
    print knn_vects
