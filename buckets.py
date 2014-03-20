#coding: utf-8
from storage import PickleStorage
from bitarray import bitarray
import numpy as np

class KlshBucket(object):

    def __init__(self, **kw):
        for key, value in kw.iteritems():
            setattr(self, key, value)

        self.buckets = {}
        self.index = []
        self.storage = PickleStorage()

    def insert_buckets(self, vector, hashed_array):
        index = self.index

        hashed_temp = np.array(hashed_array,dtype=np.int8,ndmin=1)
        hashed_temp = hashed_temp.tolist()
        hashed_list = hashed_temp[0]

        hashed = "".join(map(str, hashed_list))
        self.index = sorted(set(index + [hashed]))
        buckets = self.buckets
        if hashed not in buckets:
            buckets[hashed] = []
        buckets[hashed].append(vector)

    def select_knn(self, k, hashed_array):
        #query_vector_tuple = tuple(query_vector)
        knn_result = [] #[element->{'distance':hamming_dist,'vector':vector}]

        hashed_temp = np.array(hashed_array,dtype=np.int8,ndmin=1)
        hashed_temp = hashed_temp.tolist()
        hashed_list = hashed_temp[0]

        hashed = "".join(map(str, hashed_list))
        buckets = self.buckets
        indexes = self.index

        for index_val in indexes:
            ham_dist = self._ham_dist(hashed,index_val)
            if len(knn_result) < k:
                knn_result.append({'ham_dist':ham_dist,'vector':buckets[index_val]})
                continue

            knn_result.sort(key=lambda x:x['ham_dist'])#对ham_dist进行升序排列
            maximun_dist = knn_result[2]['ham_dist']
            if ham_dist < maximun_dist:
                knn_result[2] = {'ham_dist':ham_dist,'vector':buckets[index_val]}

        return knn_result

    def store_buckets(self):
        fw1 = open("buckets.data",'wb')
        self.storage.save(self.buckets,fw1)
        fw1.close()

        fw2 = open("buckets_index.data",'wb')
        self.storage.save(self.index,fw2)
        fw2.close()

    def load_buckets(self):
        fr1 = open('buckets.data')
        self.buckets = self.storage.load(fr1)
        fr1.close()

        fr2 = open('buckets_index.data')
        self.index = self.storage.load(fr2)
        fr2.close()

    def _ham_dist(self, hashval1, hashval2):
        xor_result = bitarray(hashval1) ^ bitarray(hashval2)
        return xor_result.count()


