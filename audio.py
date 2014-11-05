from __future__ import division
import unittest
import numpy as np
from copy import copy
import numpy as np
from scipy.spatial import ConvexHull

TEST_DIMENSIONS = 10

class TestSpatialAudio(unittest.TestCase):
    def test_create_spatialaudio(self):
        a = SpatialAudio()

    def test_coordinates(self):
        a = SpatialAudio()
        a.coordinates = Coordinates()
class TestAudio(unittest.TestCase):
    def test_Audio(self):
        a = Audio()
        
    def test_Audio_equal(self):
        a = Audio()
        a.time = np.random.rand(TEST_DIMENSIONS)
        b = Audio()
        b.freq = a.freq
        assert np.allclose(a.time, b.time)
    def test_timeVector_even(self):
        a = Audio()
        a.time = np.random.rand(10)
        a.samplingRate = 10
        assert np.allclose(a.timeVector, np.linspace(0,1,10,endpoint=False))
    def test_timeVector_odd(self):
        a = Audio()
        a.time = np.random.rand(11)
        a.samplingRate = 10
        assert np.allclose(a.timeVector, np.linspace(0,1.1,11,endpoint=False))
    def test_freqVector_even(self):
        a = Audio()
        a.time = np.random.rand(10)
        a.samplingRate = 10
        assert np.allclose(a.freqVector, np.linspace(0,5,6,endpoint=True))
    def test_freqVector_odd(self):
        a = Audio()
        a.time = np.random.rand(11)
        a.samplingRate = 10
        assert np.allclose(a.freqVector, np.linspace(0,5.5,6,endpoint=False))
    def test_get_time(self):
        a = Audio()
        a.time = np.random.rand(10)
        a.samplingRate = 10
        print a.get_time(0.5)
        assert np.allclose(a.get_time(0.5), a.time[5])
    def test_get_freq(self):
        a = Audio()
        a.time = np.random.rand(10)
        a.sync()
        a.samplingRate = 10
        assert np.allclose(a.get_freq(3.), a.freq[3])
        
class TestFFT(unittest.TestCase):
    def test_fft(self):
        a = Time()
        a.data = np.random.rand(TEST_DIMENSIONS)
        a2 = a.fft()
        # assert type(a2) == Freq  # works only in python3
    def test_fft_lengths(self):
        a = Time()
        a.data = np.random.rand(TEST_DIMENSIONS)
        b = a.fft()

    def test_fft_even(self):
        a = Time()
        a.data = np.random.rand(TEST_DIMENSIONS)
        print(a.nSamples)
        c = a.fft()
        print(c.nSamples)
        b = a.fft().ifft()
        assert np.allclose(a.data, b.data)

    def test_fft_odd(self):
        a = Time()
        a.data = np.random.rand(TEST_DIMENSIONS + 1)
        b = a.fft().ifft()
        assert np.allclose(a.data, b.data)

    def test_fft_power_even(self):
        a = Time()
        a.fft = a.fft_power
        a.data = np.random.rand(TEST_DIMENSIONS)
        b = a.fft().ifft()
        assert np.allclose(a.data, b.data)
    
    
    def test_fft_power_odd(self):
        a = Time()
        a.fft = a.fft_power
        a.data = np.random.rand(TEST_DIMENSIONS + 1)
        b = a.fft().ifft()
        assert np.allclose(a.data, b.data)
    
    
        
    def test_sync_time(self):
        a = Audio()
        a.time = np.random.rand(TEST_DIMENSIONS)
        assert a._isValidTime
        a.sync()
        assert a._isValidFreq and a._isValidTime
    def test_sync_freq(self):
        a = Audio()
        a.freq = np.random.rand(TEST_DIMENSIONS/2 + 1)
        assert a._isValidFreq
        a.sync()
        assert a._isValidTime and a._isValidFreq
class TestTimeFreq(unittest.TestCase):
    def test_timeObj(self):
        a = Time()
    def test_freqObj(self):
        a = Freq()

    def test_timeObj_data(self):
        a = Time()
        t = np.random.rand(TEST_DIMENSIONS)
        a.data = t
        assert (a.data == t).all()

    def test_freqObj_data(self):
        a = Freq()
        f = np.random.rand(TEST_DIMENSIONS)
        a.data = f
        assert (a.data == f).all()

    def test_timeObj_nSamples(self):
        a = Time()
        a.data = np.random.rand(TEST_DIMENSIONS)
        assert a.nSamples == TEST_DIMENSIONS


                
        
class TestTransform(unittest.TestCase):
    def test_transforms(self):
        x1, y1, z1 = np.random.randn(3)
        x2, y2, z2 = spherical_to_cartesian(*cartesian_to_spherical(x1, y1, z1))
        self.assertAlmostEqual(x1, x2)
        self.assertAlmostEqual(y1, y2)
        self.assertAlmostEqual(z1, z2)
        
class TestCoordinates(unittest.TestCase):
    
    def test_createInstance(self):
        c = Coordinates()

    def test_parse_input_as_cart(self):
        random = np.random.randn(10, 3)
        c = Coordinates(random)
        assert np.allclose(c.cart, random)


    def test_shape(self):
        c = Coordinates()
        c.cart = np.random.rand(10, 3)
        assert c.cart.shape == (10, 3)

    def test_shape_sph(self):
        c = Coordinates()
        c.cart = np.random.rand(10, 3)
        assert c.sph.shape == (10, 3)


    def test_shape_x(self):
        c = Coordinates()
        c.cart = np.random.rand(10, 3)
        assert c.x.shape == (10,)


    def test_xyz(self):
        c = Coordinates()
        c.cart = np.random.rand(10, 3)
        cart = np.column_stack((c.x, c.y, c.z))
        assert np.allclose(cart, c.cart)


    def test_writeCart(self):
        c = Coordinates()
        tmp = np.random.rand(10, 3)
        c.cart = tmp
        a, b = 5, 2
        assert c.cart[a,b] == tmp[a, b]


    def test_transform(self):
        c = Coordinates()
        tmp = np.random.rand(10, 3)
        c.cart = tmp
        sph = c.sph
        c.sph = sph
        a, b = 4, 0
        self.assertAlmostEqual(c.cart[a, b], tmp[a, b])


    def test_conversions_r(self):
        c = Coordinates()
        c.cart = np.random.rand(10, 3)
        r_squared = (c.cart**2).sum(axis=1)
        assert np.allclose(c.r**2, r_squared)


    def test_clone_data(self):
        c = Coordinates(np.random.randn(20, 3))
        c2 = Coordinates(c.cart)
        c.r += 10
        assert np.allclose(c.r, c2.r + 10)


    def test_phi_correct_conventions(self):
        c = Coordinates(np.random.randn(20, 3))
        test = np.all(0 <= c.phi) and np.all(c.phi < 2*np.pi)
        assert test


    def test_calculate_with_single_numbers(self):
        c = Coordinates(np.random.randn(5, 3))
        x = c.x
        print x
        c.r = c.r * 2
        print 2*x
        print c.x 
        assert np.allclose(2*x, c.x)


    def test_new_size_cart(self):
        c = Coordinates(np.random.randn(5, 3))
        c.cart = np.concatenate((c.cart,c.cart))


    def test_new_size_sph(self):
        c = Coordinates(np.random.randn(5, 3))
        c.sph = np.concatenate((c.sph,c.sph))

    def test_import_list_of_points(self):
        points = [[0,1,0],[1,1,1],[-1,-1,0],[-1,0,0]]
        c = Coordinates(points)

    def test_get_nPoints(self):
        c = Coordinates(np.random.randn(5, 3))
        n = c.nPoints
        assert n == 5

    def test_convex_hull(self):
        # h5 = AudioHDF5
        # from itaCoordinates import Coordinates
        c = Coordinates([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
        c.update_simplices()
        assert c.simplices.shape == (8,3)

    # def test_wrong_input():
    #     # todo
    #     pass

    def test_input_cart(self):
        pass


def cartesian_to_spherical(x, y, z, degrees=False):
    xsq, ysq, zsq = x*x, y*y, z*z
    r = (xsq + ysq + zsq)**0.5
    t = np.arctan2((xsq + ysq)**0.5, z)
    p = np.arctan2(y, x)
    if degrees:
        t, p = np.degrees(t), np.degrees(p)
    return r,t,p

def spherical_to_cartesian(r, t, p, degrees=False):
    if degrees:
        t, p = np.radians(t), np.radians(p)
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return x, y, z


class Coordinates(object):
    """
    This class manages a list of coordinates.
    """
    # _cart = np.ndarray((0, 3))
    simplices = None
    
    def cart2sph(self):
        r, t, p = cartesian_to_spherical(self.x, self.y, self.z)
        # use our conventions, phi from 0 to 2*pi
        p = np.mod(p, 2*np.pi)
        return r, t, p

    def sph2cart(self, r, t, p):
        x, y, z = spherical_to_cartesian(np.ravel(r), np.ravel(t), np.ravel(p))
        self.cart = np.vstack((x, y, z)).T

    def __init__(self, cart=np.ndarray((0,3))):
        self._cart = np.array(cart)
    #     try:
    #         cart = cart.cart
    #     except:
    #         pass
    #
    #     if type(cart) is 'itaCoordinates.Coordinates':
    #         cart = cart.cart
    #
    #     if cart is None:
    #         if sph is not None:
    #             self.cart = np.zeros_like(sph)
    #             self.sph = sph
    #         else:
    #             pass
    #             # print 'no coordinate data given'
    #     else:
    #         self.cart = np.array(cart)

    def _get_cart(self):
        return self._cart
    def _set_cart(self, value):
        # reshape if the input is a 1d numpy array
        self._cart = value.reshape(-1,3)
    cart = property(_get_cart, _set_cart)
    
    def _get_x(self):
        return self.cart[:, 0]
    def _set_x(self, value):
        self.cart[:, 0] = value
    x = property(_get_x, _set_x)
    
    def _get_y(self):
        return self.cart[:,1]
    def _set_y(self, value):
        self.cart[:, 1] = value
    y = property(_get_y, _set_y)
    
    def _get_z(self):
        return self.cart[:, 2]
    def _set_z(self, value):
        self.cart[:, 2] = value
    z = property(_get_z, _set_z)
    
    def _get_r(self):
        r, t, p = self.cart2sph()
        return r
    def _set_r(self, value):
        r, t, p = self.cart2sph()
        r = np.array(value).ravel()
        self.sph2cart(r, t, p)
    r = property(_get_r, _set_r)
    
    def _get_theta(self):
        r, t, p = self.cart2sph()
        return t
    def _set_theta(self, value):
        r, t, p = self.cart2sph()
        t = np.array(value).ravel()
        self.sph2cart(r, t, p)
    theta = property(_get_theta, _set_theta)
    
    def _get_theta_deg(self):
        return self.theta * 180 / np.pi
    def _set_theta_deg(self, value):
        self.theta = value * np.pi / 180
    theta_deg = property(_get_theta_deg, _set_theta_deg)
    
    def _get_phi(self):
        r, t, p = self.cart2sph()
        return p
    def _set_phi(self, value):
        r, t, p = self.cart2sph()
        p = np.array(value).ravel()
        self.sph2cart(r, t, p)
    phi = property(_get_phi, _set_phi)
    
    def _get_phi_deg(self):
        return self.phi * 180 / np.pi
    def _set_phi_deg(self, value):
        self.phi = value * np.pi / 180
    phi_deg = property(_get_phi_deg, _set_phi_deg)
    
    def _get_sph(self):
        r, t, p = self.cart2sph()
        return np.vstack((r, t, p)).T
    def _set_sph(self, value):
        # reshape if the input is a 1d numpy array
        value = value.reshape(-1,3)
        if value.shape[0] != self.cart.shape[0]:
            # size does not match, initialize with zeros
            self.cart = np.zeros_like(value)
        try:
            r = value[:, 0]
            t = value[:, 1]
            p = value[:, 2]
            self.sph2cart(r, t, p)
        except IndexError:
            # still no data here, do nothing
            print('do nothing, no data here')
    sph = property(_get_sph, _set_sph)

    def _get_nPoints(self):
        return self.cart.shape[0]
    def _set_nPoints(self, value):
        self.cart = np.zeros((value,3))
    nPoints = property(_get_nPoints, _set_nPoints)

    def update_simplices(self):
        #simplices = np.ndarray((0,3))
        THETA_LIMIT = 10 / 180. * np.pi  # in rad
        DEVIATION_PERCENT = np.cos(THETA_LIMIT)
        
        if self.cart.shape[1] == 3 and self.cart.size > 0:

            # link: http://www.qhull.org/html/qh-optq.htm#QbB
            simplices = ConvexHull(points=self.cart, qhull_options='QbB').simplices

            grid = self.pull_on_unit_sphere()

            max_z = np.max(grid.z)
            min_z = np.min(grid.z)

            # open only gaps which are large
            if max_z > DEVIATION_PERCENT: max_z = 1.
            if min_z < -DEVIATION_PERCENT: min_z = -1.

            is_z_max = np.isclose(grid.z[simplices], max_z)
            is_z_min = np.isclose(grid.z[simplices], min_z)
            #is_z_min = self.z[simplices] == np.min(self.z)

            is_valid_min = np.sum(is_z_min, axis=1) < 3
            is_valid_max = np.sum(is_z_max, axis=1) < 3
            is_valid = np.logical_and(is_valid_min, is_valid_max)
            self.simplices = simplices[is_valid,:]

    def pull_on_unit_sphere(self):
        grid = copy(self)
        grid.cart = grid.cart / self.r[:, None] # pull on unit sphere
        return grid


class Time(object):
    data = np.ndarray((0,0))
    # samplingRate = 44100
    def _isEvenTimeDomain():
        return(not(data % 2))
    @property
    def nSamples(self):
        n = self.data.shape[-1]
        return n
        
    @property
    def nBins(self):
        return (self.nSamples / 2) + 1

    @property
    def timeVector(self):
        if self._isEvenTimeDomain:
            # even samples
            return np.linspace(0, self.samplingRate/2., self.nBins)
        else:
            print 'check me: odd nr of samples, freqVector'
            return np.linspace(0, self.samplingRate/2. * (1 - 1/(2.*self.nBins)), self.nBins)
    def fft(self):
        return self.fft_energy()

    def fft_energy(self):
        a = Freq()
        a.ifft = a.ifft_energy
        a.data = np.fft.rfft(self.data)
        a._isEvenTimeDomain = not (self.nSamples % 2)
        return a
    
    def fft_power(self):
        a = Freq()
        a.ifft = a.ifft_power
        a.data = np.fft.rfft(self.data) / self.nSamples
        a.data[...,1:] *= 2**0.5
        a.data[...,-1] /= 2
        a._isEvenTimeDomain = not (self.nSamples % 2)
        return a
            


class Freq(object):    
    data = np.ndarray((0,0))
    # samplingRate = 44100
    _isEvenTimeDomain = True
    @property
    def nBins(self):
        return self.data.shape[-1]
        
    @property
    def nSamples(self):
        if self._isEvenTimeDomain:
            return 2 * (self.nBins - 1)
        else:
            return (2 * self.nBins - 1)
        
    @property
    def freqVector(self):
        if self._isEvenTimeDomain:
            # even samples
            return np.linspace(0, self.samplingRate/2., self.nBins)
        else:
            print 'check me: odd nr of samples, freqVector'
            return np.linspace(0, self.samplingRate/2. * (1 - 1/(2.*self.nBins)), self.nBins)
    def ifft(self):
        return self.ifft_energy()
        
    def ifft_energy(self):
        a = Time()
        a.fft = a.fft_energy
        a.data = np.fft.irfft(self.data, self.nSamples)
        return a
    
    def ifft_power(self):
        a = Time()
        a.fft = a.fft_power
        f = self.data
        f[...,1:] /= 2**0.5
        f[...,-1] *= 2
        a.data = np.fft.irfft(f, self.nSamples) * self.nSamples
        return a
    
    def _get_freqVector(self):
        if np.mod(self.time.shape[0],2):
            # even samples
            return np.linspace(0, self.samplingRate/2., self.nBins)
        else:
            return np.linspace(0, self.samplingRate/2. * (1 - 1/(2.*self.nBins)), self.nBins)
            print 'check me: odd nr of samples, freqVector'
            #return None
    freqVector = property(_get_freqVector)
    
class Audio(object):
    
    samplingRate = 44100.
        
    _isValidTime = False
    _isValidFreq = False
    
    _timeObj = Time()
    _freqObj = Freq()
    
    def sync(self):
        if not self._isValidTime and self._isValidFreq:
            self._timeObj = self._freqObj.ifft()
            self._isValidTime = True
        if not self._isValidFreq and self._isValidTime:
            self._freqObj = self._timeObj.fft()
            self._isValidFreq = True
            
    @property
    def time(self):
        if not self._isValidTime:
            if self._isValidFreq:
                self._timeObj = self._freqObj.ifft()
                self._isValidTime = True
        return self._timeObj.data
            
    @time.setter
    def time(self, data):
        self._timeObj.data = data
        self._isValidTime = True
        self._isValidFreq = False
        
    @property
    def freq(self):
        if not self._isValidFreq:
            if self._isValidTime:
                self._freqObj = self._timeObj.fft()
                self._isValidFreq = True
        return self._freqObj.data
    @freq.setter
    def freq(self, data):
        self._freqObj.data = data
        self._isValidFreq = True
        self._isValidTime = False
    
    @property
    def freqVector(self):
        if not self._isValidFreq:
            self.sync()
        if self._isValidFreq:
            if self._freqObj._isEvenTimeDomain:
                return np.linspace(0, self.samplingRate/2., self._freqObj.nBins)
            else:
                return np.linspace(0, self.samplingRate/2. * (1 - 1/(2.*self._freqObj.nBins)), self._freqObj.nBins)

    @property
    def timeVector(self):
        if not self._isValidTime:
            self.sync()
        if self._isValidTime:
            return np.linspace(0, self._timeObj.nSamples / self.samplingRate, self._timeObj.nSamples, endpoint=False)

    def get_time(self, time):
        index = np.argmin(np.abs(self.timeVector - time))
        return self.time[...,index]
        
    def get_freq(self, freq):
        index = np.argmin(np.abs(self.freqVector - freq))
        return self.freq[...,index]


class SpatialAudio(Audio):
    _coordinates = Coordinates()
    name = 'Description of the data'
    
    @property
    def coordinates(self):
        return self._coordinates
    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value
        self._coordinates.update_simplices()

