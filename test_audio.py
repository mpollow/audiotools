import unittest
from audio import *

TEST_DIMENSIONS = (2, 10)
# last dimension is time/freq

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
        # TODO: check me, _get_time should not be called here
        assert np.allclose(a.time[:,ch,:], a._get_time(slice(None),ch))

    def test_compare_point(self):
        a = Audio()
        p = 7
        a.time = np.random.rand(10,20,30)
        # print a.time[p,:,:].shape
        # print a.get_time(point=p).shape
        assert np.allclose(a.time[p,:,:], a._get_time(slice(None), slice(None), p))


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
        assert np.allclose(a._get_time(5), a.time[...,5])
    def test_get_time_float(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a.samplingRate = 10
        assert np.allclose(a._get_time(0.5), a.time[...,5])
    def test_get_freq(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a._sync()
        a.samplingRate = 10
        assert np.allclose(a._get_freq(3), a.freq[...,3])
    def test_get_freq_float(self):
        a = Audio()
        a.time = np.random.rand(1,10)
        a._sync()
        a.samplingRate = 10
        assert np.allclose(a._get_freq(3.), a.freq[...,3])

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
        dim = list(TEST_DIMENSIONS)
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
        dim = list(TEST_DIMENSIONS)
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
        dim = list(TEST_DIMENSIONS)
        dim[-1] //= 2
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
        dim = list(TEST_DIMENSIONS)
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


if __name__ == '__main__':
    unittest.main()
