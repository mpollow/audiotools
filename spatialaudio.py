from audio import Audio
from coordinates import Coordinates

import numpy as np
from copy import copy
from scipy.spatial import ConvexHull

        
class SpatialAudio(Audio):
    coordinates = Coordinates()

class SOFA_Audio(SpatialAudio):
    
    def __init__(self, filename):
        import h5py
        h5 = h5py.File(filename)
        # [ind for ind in h5.iterkeys()]
        self.filename = filename
        self.samplingRate = h5['Data.SamplingRate'][:]
        self.time = h5['Data.IR'][:]        
        self.coordinates = self._position2coordinates(h5['SourcePosition'])
        self._sync()
        # vis.BalloonGUI()

    def _position2coordinates(self, pos):
        p = pos[...,0] / 180 * np.pi
        t = (90 - pos[...,1]) / 180 * np.pi
        r = pos[...,2]
        c = Coordinates()
        c.sph = np.vstack((r,t,p)).T
        return c
