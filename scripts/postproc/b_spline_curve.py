"""Helper class for 3D B-spline curves.

Defines a minimalistic class for spatial curves defined in terms of B-splines.
Curves can be created from CHAP output files.
"""


import warnings

import numpy as np
from scipy.interpolate import BSpline as bspl
from scipy.optimize import root


class BSplineCurve:
    """
    B-Spline spatial curve creatable from CHAP output data.
    """

    def __init__(self, chap_data):
        """
        Create a spline curve from CHAP data.
        """

        # extract spline knots:
        self._knots = np.array(chap_data["molPathCentreLineSpline"]["knots"])

        # create knot vector with repeated ends:
        self._degree = 3
        self._t = np.concatenate(
            np.repeat(self._knots[0], self._degree-1),
            self._knots, np.repeat(self._knots[-1], self._degree-1)
        )

        # create nd-array representing control point coordinates:
        self._ctrlX = np.array(chap_data["molPathCentreLineSpline"]["ctrlX"])
        self._ctrlY = np.array(chap_data["molPathCentreLineSpline"]["ctrlY"])
        self._ctrlZ = np.array(chap_data["molPathCentreLineSpline"]["ctrlZ"])

        # perform b-spline interpolation along each axis:
        self._bsplX = bspl(self._t, self._ctrlX, self._degree, extrapolate=True)
        self._bsplY = bspl(self._t, self._ctrlY, self._degree, extrapolate=True)
        self._bsplZ = bspl(self._t, self._ctrlZ, self._degree, extrapolate=True)

        # make sure spline coordinate is strictly monotonic:
        strictly_increasing = np.all(
            self._bsplZ(np.unique(self._knots))[1:]
            <= self._bsplZ(np.unique(self._knots))[:-1],
            axis=0
        )
        strictly_decreasing = np.all(
            self._bsplZ(np.unique(self._knots))[1:]
            >= self._bsplZ(np.unique(self._knots))[:-1],
            axis=0
        )
        if not strictly_decreasing ^ strictly_increasing:
            warnings.warn(
                "Spline curve z-coordinate is not strictly monotonic. May be"
                " unable to convert z-coordinate to s-coordinate."
            )

    def root_fun(self, s, z0):
        """
        Function definiting the root of z(s).
        """

        return(self._bsplZ(s) - z0)

    def find_root(self, z0):
        """
        Solves root finding problem.
        """

        r = root(
            fun=self.root_fun,
            x0=0.0,
            args=(z0),
            tol=1e-8,
            method="hybr"
        )

        return(r.x[0])

    def z2s(self, z):
        """
        Convert given z-coordinate to s-coordinate.
        """

        # vectorize root finding function:
        f = np.vectorize(self.find_root)

        # apply to all input arguments:
        return(f(z))
