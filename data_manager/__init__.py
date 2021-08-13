from .Mars import *
from .Duke import *
from .PRID import *
from .iLIDSVID import *

__factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
    'duke':DukeMTMCVidReID
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)