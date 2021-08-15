from __future__ import absolute_import

from models.baseline import ResNet50
from models.PSTA import PSTA

__factory = {
    'ResNet50' : ResNet50,
    'PSTA' : PSTA,
}

def get_names():
    return __factory.keys()

def init_model(name,*args,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model :{}".format(name))
    return __factory[name](*args,**kwargs)