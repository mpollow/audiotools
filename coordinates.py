from __future__ import division
import numpy as np
from numpy import pi
from copy import copy
from scipy.spatial import ConvexHull

def cartesian_to_spherical(x, y, z, degrees=False):
    """
    Converts Cartesian coordinates to spherical coordinates
    cart = (x, y, z)
    sph = (r, theta, phi)   [Rad]

    >>> cartesian_to_spherical(1.0, 0.0, 0.0)
    (1.0, 1.5707963267948966, 0.0)
    >>> cartesian_to_spherical(0.0, 1.0, 0.0)
    (1.0, 1.5707963267948966, 1.5707963267948966)
    >>> cartesian_to_spherical(0.0, 0.0, 1.0)
    (1.0, 0.0, 0.0)
    """
    xsq, ysq, zsq = x*x, y*y, z*z
    r = (xsq + ysq + zsq)**0.5
    t = np.arctan2((xsq + ysq)**0.5, z)
    p = np.arctan2(y, x)
    if degrees:
        t, p = np.degrees(t), np.degrees(p)
    return r, t, p

def spherical_to_cartesian(r, t, p, degrees=False):
    if degrees:
        t, p = np.radians(t), np.radians(p)
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return x, y, z


class Coordinates:
    """
    This class manages a list of coordinates.
    """
    
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
        self.simplices = None

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


if __name__=='__main__':
    import doctest
    doctest.testmod()
