#coding: utf-8
try:
    import cPickle as pickle
except ImportError:
    import pickle


class PickleStorage(object):

    def save(self, data, file_path):
        pickle.dump(data, file_path)

    def load(self, file_path):
        return pickle.load(file_path)