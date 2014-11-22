from __future__ import division
import unittest
import numpy as np
from copy import copy
from scipy.spatial import ConvexHull

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
    """
    Implementation of time domain audio data.
    """
    data = np.ndarray((0,0))
    def _isEvenTimeDomain():
        return(not(data % 2))
    def nSamples(self):
        return self.data.shape[-1]
    def nBins(self):
        return (self.nSamples() // 2) + 1

    def timeValues(self):
        if self._isEvenTimeDomain:
            return np.linspace(0, self.samplingRate/2., self.nBins())
        else:
            # 'check me: odd nr of samples, freqValues'
            return np.linspace(0, self.samplingRate/2. * (1 - 1/(2.*self.nBins())), self.nBins())
    def fft(self):
        return self.fft_energy()

    def fft_energy(self):
        freq = Freq()
        if self.data.size:
            freq.data = np.fft.rfft(self.data)        
        freq.ifft = freq.ifft_energy
        freq._isEvenTimeDomain = not (self.nSamples() % 2)
        return freq
    
    def fft_power(self):
        freq = self.fft_energy()
        freq.ifft = freq.ifft_power
        freq.data /= self.nSamples()
        freq.data[...,1:] *= 2**0.5
        freq.data[...,-1] /= 2
        return freq


class Freq(object):    
    """
    Implementation of frequency domain audio data.
    """
    data = np.ndarray((0,0))
    _isEvenTimeDomain = True
    def nBins(self):
        return self.data.shape[-1]
        
    def nSamples(self):
        if self._isEvenTimeDomain:
            return 2 * (self.nBins() - 1)
        else:
            return (2 * self.nBins() - 1)
        
    def freqValues(self):
        if self._isEvenTimeDomain:
            # even samples
            return np.linspace(0, self.samplingRate/2., self.nBins())
        else:
            print 'check me: odd nr of samples, freqValues'
            return np.linspace(0, self.samplingRate/2. * (1 - 1/(2.*self.nBins())), self.nBins())
    def ifft(self):
        return self.ifft_energy()
        
    def ifft_energy(self):
        time = Time()
        if self.data.size:
            time.data = np.fft.irfft(self.data, self.nSamples())
        time.fft = time.fft_energy
        return time
    
    def ifft_power(self):
        freq = self.data
        freq *= self.nSamples()
        freq[...,1:] /= 2**0.5
        freq[...,-1] *= 2
        time = self.ifft_energy()
        time.fft = time.fft_power
        return time
    
class Audio(object):
    """
    This class stores audio data in time and frequency domain. The FFT or IFFT is applied as needed.
    """
    # samplingRate = 44100.
    # current = 0
        
    _isValidTime = False
    _isValidFreq = False
    
    _timeObj = Time()
    _freqObj = Freq()
    
    def __init__(self):
        self.samplingRate = 44100.
        self.indexposition = 0
        
    def _sync(self):
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
        return self.get_time()            
    @time.setter
    def time(self, data):
        self.set_time(data)

    @property
    def freq(self):
        if not self._isValidFreq:
            if self._isValidTime:
                self._freqObj = self._timeObj.fft()
                self._isValidFreq = True
        return self.get_freq()

    @freq.setter
    def freq(self, data):
        self.set_freq(data)


    def get_time(self, *args, **kwargs):
        data = self._timeObj.data
        index = [slice(None) for i in range(data.ndim)]
        # convert time to index for float input
        for ind, arg in enumerate(args):
            if ind is 0 and type(arg) is float:
                arg = self.timeValues(arg)
            try:
                index[-1-ind] = arg
            except:
                pass
        return data[index]
        
    def get_freq(self, *args, **kwargs):
        data = self._freqObj.data
        index = [slice(None) for i in range(data.ndim)]
        # convert frequency to index for float input
        for ind, arg in enumerate(args):
            if ind is 0 and type(arg) is float:
                arg = self.freqValues(arg)
            try:
                index[-1-ind] = arg
            except:
                pass
        return data[index]

    def set_time(self, data, *args, **kwargs):
        self._timeObj.data = np.array(data)
        self._isValidTime = True
        self._isValidFreq = False
        
    def set_freq(self, data, *args, **kwargs):
        self._timeObj.data = np.array(data)
        self._isValidFreq = True
        self._isValidTime = False

    def freqValues(self, value=None):
        if not self._isValidFreq:
            self._sync()
        if self._isValidFreq:
            if self._freqObj._isEvenTimeDomain:
                linindex = np.linspace(0, self.samplingRate/2., self.nBins)
            else:
                linindex = np.linspace(0, self.samplingRate/2. * (1 - 1/(2.*self.nBins)), self.nBins)
        if value:
            return (np.abs(linindex - value)).argmin()
        else:
            return linindex
            
            
    def timeValues(self, value=None):
        if not self._isValidTime:
            self._sync()
        if self._isValidTime:
            linindex = np.linspace(0, self.nSamples / self.samplingRate, self.nSamples, endpoint=False)
        if value:
            return np.abs(linindex - value).argmin()
        else:
            return linindex

    @property
    def nSamples(self):
        if self._isValidTime:
            n = self._timeObj.nSamples()
        elif self._isValidFreq:
            n = self._freqObj.nSamples()
        else:
            n = 0
        return n
    @property
    def nBins(self):
        n = [0]  # default
        if self._isValidTime:
            n.append(self._timeObj.nBins())
        if self._isValidFreq:
            n.append(self._freqObj.nBins())
        if n.__len__() == 3:
            # checks for identical results, independent of domain
            assert n[1] == n[2]
            return n[1]
        n = n[::-1]  # revert list
        return int(n[0]) # use first entry (nBins or 0)
        
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

## TEST FUNCTIONS
TEST_DIMENSIONS = [2,10]
class TestSOFA_Audio(unittest.TestCase):
    def test_load(self):
        sofa = SOFA_Audio('/Volumes/Macintosh_HD/Users/pollow/Projekte/Python/audiotools/test.sofa')

    def test_more(self):
        pass

class TestSpatialAudio(unittest.TestCase):
    def test_create_spatialaudio(self):
        a = SpatialAudio()

    def test_coordinates(self):
        a = SpatialAudio()
        a.coordinates = Coordinates()

    def test_time_channel(self):
        a = SpatialAudio()
        ch = 1
        a.coordinates = Coordinates()
        a.time = np.random.rand(10,5,3)
        time = a.get_time(slice(None),ch)
        assert np.allclose(time, a.time[:,ch,:])

class TestAudioIndexing(unittest.TestCase):

    def test_size_correct(self):
        a = Audio()
        a.time = np.random.rand(4,8,16)

    def test_compare_channel(self):
        a = Audio()
        ch = 4
        a.time = np.random.rand(5,10,15)
        # print a.time[:,ch,:].shape
        # print a.get_time(channel=ch).shape
        assert np.allclose(a.time[:,ch,:], a.get_time(slice(None),ch))

    def test_compare_point(self):
        a = Audio()
        p = 7
        a.time = np.random.rand(10,20,30)
        # print a.time[p,:,:].shape
        # print a.get_time(point=p).shape
        assert np.allclose(a.time[p,:,:], a.get_time(slice(None), slice(None), p))


class TestAudio(unittest.TestCase):

    def test_Audio(self):
        a = Audio()

    def test_Audio_equal(self):
        a = Audio()
        a.time = np.random.rand(*TEST_DIMENSIONS)
        b = Audio()
        b.time = a.time
        assert np.allclose(a.freq, b.freq)

    def test_timeValues_even(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a.samplingRate = 10
        assert np.allclose(a.timeValues(), np.linspace(0,1,10,endpoint=False))
    def test_timeValues_odd(self):
        a = Audio()
        a.time = np.random.rand(1,11)
        a.samplingRate = 10
        assert np.allclose(a.timeValues(), np.linspace(0,1.1,11,endpoint=False))
    def test_freqValues_even(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a.samplingRate = 10
        assert np.allclose(a.freqValues(), np.linspace(0,5,6,endpoint=True))
    def test_freqValues_odd(self):
        a = Audio()
        a.time = np.random.rand(1,11)
        a.samplingRate = 10
        assert np.allclose(a.freqValues(), np.linspace(0,5.5,6,endpoint=False))
    def test_get_time(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a.samplingRate = 10
        assert np.allclose(a.get_time(5), a.time[...,5])
    def test_get_time_float(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a.samplingRate = 10
        assert np.allclose(a.get_time(0.5), a.time[...,5])
    def test_get_freq(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a._sync()
        a.samplingRate = 10
        assert np.allclose(a.get_freq(3), a.freq[...,3])
    def test_get_freq_float(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a._sync()
        a.samplingRate = 10
        assert np.allclose(a.get_freq(3.), a.freq[...,3])

    def test_get_nSamples(self):
        a = Audio()
        a.time = np.random.rand(*TEST_DIMENSIONS)
        assert a.nSamples == TEST_DIMENSIONS[-1]

class TestTime(unittest.TestCase):
    """
    Class for time domain data.
    """  
    def test_time(self):
        time = Time()
    def test_time_data(self):
        time = Time()
        time.data = np.random.randn(2,10)
        assert time.data.shape == (2,10)
    def test_time_data_list(self):
        data = [1.,0.,0.,0.]
        time = Time()
        time.data = np.array(data)
        assert np.all(time.data == data)
    def test_time_nSamples(self):
        time = Time()
        time.data = np.random.randn(2,10)
        assert time.nSamples() == 10
    def test_time_nBins(self):
        time = Time()
        time.data = np.random.randn(2,10)
        assert time.nBins() == 6
    def test_time_fft(self):
        time = Time()
        freq = time.fft()
        assert type(freq) is Freq
    def test_time_fft_example(self):
        time = Time()
        time.data = np.array([1.,0.,0.,0.])
        freq = time.fft()
        assert np.allclose(freq.data, [1.,1.,1.])

class TestFreq(unittest.TestCase):  
    """
    Class for frequency domain data.
    """  
    def test_freq(self):
        freq = Freq()
    def test_freq_data(self):
        freq = Freq()
        freq.data = np.random.randn(2,9)
        assert freq.data.shape == (2,9)
    def test_freq_data_list(self):
        data = [1.,0.,0.,0.]
        freq = Freq()
        freq.data = np.array(data)
        assert np.all(freq.data == data)
    def test_freq_nSamples(self):
        freq = Freq()
        freq.data = np.random.randn(2,9)
        assert freq.nSamples() == 16
    def test_freq_nBins(self):
        freq = Freq()
        freq.data = np.random.randn(2,9)
        assert freq.nBins() == 9
    def test_freq_ifft(self):
        freq = Freq()
        time = freq.ifft()
        assert type(time) is Time
    def test_freq_ifft_example(self):
        freq = Freq()
        freq.data = np.array([1.,1.,1.])
        time = freq.ifft()
        assert np.allclose(time.data, [1.,0.,0.,0.])


class TestFFT(unittest.TestCase):

    def test_fft(self):
        a = Time()
        a.data = np.random.rand(*TEST_DIMENSIONS)
        a2 = a.fft()
        # assert type(a2) == Freq  # works only in python3
    def test_fft_lengths(self):
        a = Time()
        a.data = np.random.rand(*TEST_DIMENSIONS)
        b = a.fft()
    def test_fft_even(self):
        a = Time()
        a.data = np.random.rand(*TEST_DIMENSIONS)
        c = a.fft()
        b = a.fft().ifft()
        assert np.allclose(a.data, b.data)
    def test_fft_odd(self):
        a = Time()
        dim = TEST_DIMENSIONS
        dim[-1] += 1
        a.data = np.random.rand(*dim)
        b = a.fft().ifft()
        assert np.allclose(a.data, b.data)
    def test_fft_power_even(self):
        a = Time()
        a.fft = a.fft_power
        a.data = np.random.rand(*TEST_DIMENSIONS)
        # print a.data
        b = a.fft().ifft()
        # print b.data
        assert np.allclose(a.data, b.data)
    def test_fft_power_odd(self):
        a = Time()
        a.fft = a.fft_power
        dim = TEST_DIMENSIONS
        dim[-1] += 1
        a.data = np.random.rand(*dim)
        b = a.fft().ifft()
        assert np.allclose(a.data, b.data)
    def test_sync_time(self):
        a = Audio()
        a.time = np.random.rand(*TEST_DIMENSIONS)
        assert a._isValidTime
        a._sync()
        assert a._isValidFreq and a._isValidTime
    def test_sync_freq(self):
        a = Audio()        
        dim = TEST_DIMENSIONS
        dim[-1] /= 2
        dim[-1] += 1
        a.freq = np.random.rand(*dim)
        assert a._isValidFreq
        a._sync()
        assert a._isValidTime and a._isValidFreq

class TestTimeFreq(unittest.TestCase):
    def test_timeObj(self):
        a = Time()
    def test_freqObj(self):
        a = Freq()
    def test_timeObj_data(self):
        a = Time()
        t = np.random.rand(*TEST_DIMENSIONS)
        a.data = t
        assert (a.data == t).all()
    def test_freqObj_data(self):
        a = Freq()
        f = np.random.rand(*TEST_DIMENSIONS)
        a.data = f
        assert (a.data == f).all()
    def test_timeObj_nSamples(self):
        a = Time()
        a.data = np.random.rand(*TEST_DIMENSIONS)
        assert a.nSamples() == TEST_DIMENSIONS[-1]
    def test_timeObj_nBins(self):
        a = Time()
        dim = TEST_DIMENSIONS
        a.data = np.random.rand(*dim)
        assert a.nBins() == dim[-1] // 2 + 1
    def test_freqObj_nSamples(self):
        a = Freq()
        dim = TEST_DIMENSIONS
        a.data = np.random.rand(*dim)
        dim[-1] = (TEST_DIMENSIONS[-1] - 1) * 2
        if not a._isEvenTimeDomain:
            dim[-1] += 1
        assert a.nSamples() == dim[-1]
    def test_freqObj_nBins(self):
        a = Freq()
        dim = TEST_DIMENSIONS
        a.data = np.random.rand(*dim)
        assert a.nBins() == dim[-1]


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
        c.r = c.r * 2
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


if __name__ == '__main__':
    unittest.main()