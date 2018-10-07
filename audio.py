from __future__ import division
import numpy as np
from copy import copy


class Time:
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


class Freq:    
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
            print('check me: odd nr of samples, freqValues')
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
    
class Audio:
    """
    This class stores audio data in time and frequency domain. The FFT or IFFT is applied as needed.
    >>> 3 + 5
    8
    """
        
    def __init__(self):
        self.samplingRate = 48000.
        self._isValidTime = False
        self._isValidFreq = False
        self._timeObj = Time()
        self._freqObj = Freq()

    def __repr__(self):
        if self._isValidTime and self._isValidFreq:
            domain = 'time/freq'
        elif self._isValidTime:
            domain = 'time'
        elif self._isValidFreq:
            domain = 'freq'
        else:
            domain = 'N/A'
        return f'Audio({domain} of {self.time.shape}@{self.samplingRate}Hz)'
    
        
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
        return self._get_time()            
    @time.setter
    def time(self, data):
        self._set_time(data)

    @property
    def freq(self):
        if not self._isValidFreq:
            if self._isValidTime:
                self._freqObj = self._timeObj.fft()
                self._isValidFreq = True
        return self._get_freq()

    @freq.setter
    def freq(self, data):
        self._set_freq(data)


    def _get_time(self, *args, **kwargs):
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
        
    def _get_freq(self, *args, **kwargs):
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

    def _set_time(self, data, *args, **kwargs):
        self._timeObj.data = np.array(data)
        self._isValidTime = True
        self._isValidFreq = False
        
    def _set_freq(self, data, *args, **kwargs):
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
       

if __name__=='__main__':
    import doctest
    doctest.testmod()
