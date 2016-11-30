"""
Implementation of ValBag class
Author: Jake Soloff
"""

import numpy as np

class ValBag(object):
    def __init__(self, **kwargs):
        ''' Create a ValBag object 
        '''
        for key, val in kwargs.iteritems():
            self.setField(key,val)

    def setField(self, key, val, dims=None):
        if not isinstance(val, np.ndarray):
            try:
                val = np.array([val]) 
            except: 
                raise Exception("Can only assign numeric types to ValBag")
        if dims is not None and dims != val.shape:
            raise Exception("Assigned value for parameter field does not match listed dimension")
        setattr(self, key, val)

    def removeField(self, key):
        val = getattr(self, key)
        delattr(self, key)
        return val
