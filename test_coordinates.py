import unittest
from coordinates import *

TEST_DIMENSIONS = (2, 10)

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
