"""
Palettes are the link between data values and the values along
the dimension of a scale. Before a collection of values can be
represented on a scale, they are transformed by a palette. This
transformation is knowing as mapping. Values are mapped onto a
scale by a palette.

Scales tend to have restrictions on the magnitude of quantities
that they can intelligibly represent. For example, the size of
a point should be significantly smaller than the plot panel
onto which it is plotted or else it would be hard to compare
two or more points. Therefore palettes must be created that
enforce such restrictions. This is the reason for the ``*_pal``
functions that create and return the actual palette functions.
"""

from dataclasses import dataclass
import numpy as np
from typing import Any


# if typing.TYPE_CHECKING:
#    from typing import Any, Optional, Sequence
#
#    from mizani.typing import (
#        Callable,
#        FloatArrayLike,
#        NDArrayFloat,
#        RGBHexColor,
#    )


# def rescale(
#    x: FloatArrayLike,
#    to: TupleFloat2 = (0, 1),
#    _from: Optional[TupleFloat2] = None,
# ) -> NDArrayFloat:
#    """
#    Rescale numeric vector to have specified minimum and maximum.
#
#    Parameters
#    ----------
#    x : array_like | numeric
#        1D vector of values to manipulate.
#    to : tuple
#        output range (numeric vector of length two)
#    _from : tuple
#        input range (numeric vector of length two).
#        If not given, is calculated from the range of x
#
#    Returns
#    -------
#    out : array_like
#        Rescaled values
#
#    Examples
#    --------
#    >>> x = [0, 2, 4, 6, 8, 10]
#    >>> rescale(x)
#    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
#    >>> rescale(x, to=(0, 2))
#    array([0. , 0.4, 0.8, 1.2, 1.6, 2. ])
#    >>> rescale(x, to=(0, 2), _from=(0, 20))
#    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
#    """
#    __from = (np.min(x), np.max(x)) if _from is None else _from
#    return np.interp(x, __from, to)


SPACE256 = np.arange(256)
INNER_SPACE256 = SPACE256[1:-1]
ROUNDING_JITTER = 1e-12


@dataclass
class GradientMap:
    colors: Any
    values: Any = None

    def __post_init__(self):
        if self.values is None:
            values = np.linspace(0, 1, len(self.colors))
        elif len(self.colors) < 2:
            raise ValueError("A color gradient needs two or more colors")
        else:
            values = np.asarray(self.values)
            if values[0] != 0 or values[-1] != 1:
                raise ValueError(
                    "Value points of a color gradient should start"
                    "with 0 and end with 1. "
                    f"Got {values[0]} and {values[-1]}"
                )

        if len(self.colors) != len(values):
            raise ValueError(
                "The values and the colors are different lengths"
                f"colors={len(self.colors)}, values={len(values)}"
            )

        colors = self.colors

        self._data = np.asarray(colors)
        self._r_lookup = interp_lookup(values, self._data[:, 0])
        self._g_lookup = interp_lookup(values, self._data[:, 1])
        self._b_lookup = interp_lookup(values, self._data[:, 2])

    def _generate_colors(self, x):
        """
        Lookup colors in the interpolated ranges

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        """
        x = np.asarray(x)
        idx = np.round((x * 255) + ROUNDING_JITTER).astype(int)
        arr = np.column_stack([self._r_lookup[idx], self._g_lookup[idx], self._b_lookup[idx]])
        return [rgb_to_hex(c) for c in arr]


def interp_lookup(x, values):
    """
    Create an interpolation lookup array

    This helps make interpolating between two or more colors
    a discrete task.

    Parameters
    ----------
    x:
        Breaks In the range [0, 1]. Must include 0 and 1 and values
        should be sorted.
    values:
        In the range [0, 1]. Must be the same length as x.
    """
    # - Map x from [0, 1] onto [0, 255] i.e. the color channel
    #   breaks (continuous)
    # - Find where x would be mapped onto the grid (discretizing)
    # - Find the distance between the discrete breaks and the
    #   continuous values of x (with each value scaled by the distance
    #   to previous x value)
    # - Expand the scaled distance (how far to move at each point) to a
    #   value, and move by that scaled distance from the previous point
    x256 = x * 255
    ind = np.searchsorted(x256, SPACE256)[1:-1]
    ind_prev = ind - 1
    distance = (INNER_SPACE256 - x256[ind_prev]) / (x256[ind] - x256[ind_prev])
    lut = np.concatenate(
        [
            [values[0]],
            distance * (values[ind] - values[ind_prev]) + values[ind_prev],
            [values[-1]],
        ]
    )
    return np.clip(lut, 0, 1)


class _continuous_pal:
    """
    Continuous palette maker
    """

    def __call__(self, x):
        """
        Palette method
        """
        ...


@dataclass
class gradient_n_pal(_continuous_pal):
    """
    Create a n color gradient palette

    Parameters
    ----------
    colors : list
        list of colors
    values : list, optional
        list of points in the range [0, 1] at which to
        place each color. Must be the same size as
        `colors`. Default to evenly space the colors

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        parameter either a :class:`float` or a sequence
        of floats maps those value(s) onto the palette
        and returns color(s). The float(s) must be
        in the range [0, 1].

    Examples
    --------
    >>> palette = gradient_n_pal(['red', 'blue'])
    >>> palette([0, .25, .5, .75, 1])
    ['#ff0000', '#bf0040', '#7f0080', '#4000bf', '#0000ff']
    >>> palette([-np.inf, 0, np.nan, 1, np.inf])
    [None, '#ff0000', None, '#0000ff', None]
    """

    colors: Any
    values: Any = None

    def __post_init__(self):
        self._gmap = GradientMap(self.colors, self.values)

    def __call__(self, x):
        return self._gmap.continuous_palette(x)
