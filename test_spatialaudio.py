from spatialaudio import *

import unittest

TEST_DIMENSIONS = (2, 10)

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
        time = a._get_time(slice(None),ch)
        assert np.allclose(time, a.time[:,ch,:])


if __name__ == '__main__':
    unittest.main()
