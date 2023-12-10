from __future__ import annotations

from .error import NumberOfElementError, OverCorrection
from .models import DataArray, NUMERICS
from .fits import Fits
from .utils import Fixer, Check

from logging import getLogger, Logger

from glob import glob

import pandas as pd
import numpy as np

from astropy.nddata import CCDData
from ccdproc import Combiner
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt, animation
from sep import Background

from typing_extensions import Self

from pathlib import Path
from typing import List, Union, Any, Optional, Iterator, Hashable, Dict

from astropy.io.fits.header import Header


class FitsArray(DataArray):
    def __init__(self, fits_list: List[Fits], logger: Optional[Logger] = None) -> None:

        self.logger = getLogger(f"{self.__class__.__name__}") if logger is None else logger

        fits_list = [
            each
            for each in fits_list
            if isinstance(each, Fits)
        ]

        if len(fits_list) < 1:
            self.logger.error("No image was provided")
            raise NumberOfElementError("No image was provided")

        self.fits_list = fits_list

    def __str__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"(@: '{id(self)}', nof:'{len(self)}')"

    def __iter__(self) -> Iterator[Fits]:
        for x in self.fits_list:
            yield x

    def __getitem__(self, key: Union[int, slice]) -> Union[Fits, FitsArray]:

        if isinstance(key, int):
            return self.fits_list[key]
        elif isinstance(key, slice):
            return FitsArray(self.fits_list[key])

        self.logger.error("Wrong slice")
        raise ValueError("Wrong slice")

    def __delitem__(self, key) -> None:
        del self.fits_list[key]

    def __len__(self) -> int:
        return len(self.fits_list)

    def __abs__(self) -> List[str]:
        return [str(fits.file.absolute()) for fits in self.fits_list]

    def __add__(self, other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]) -> Self:
        if not isinstance(other, (self.__class__, Fits, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.add(other)

    def __radd__(self, other: Union[Self, float, int, List[Union[Fits, float, int]]]) -> Self:

        if not isinstance(other, (self.__class__, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.add(other)

    def __sub__(self, other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]) -> Self:
        if not isinstance(other, (self.__class__, Fits, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.sub(other)

    def __rsub__(self, other: Union[Self, float, int, List[Union[Fits, float, int]]]) -> Self:
        if not isinstance(other, (self.__class__, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.mul(-1).add(other)

    def __mul__(self, other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]) -> Self:
        if not isinstance(other, (self.__class__, Fits, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.mul(other)

    def __rmul__(self, other: Union[Self, float, int, List[Union[Fits, float, int]]]) -> Self:
        if not isinstance(other, (self.__class__, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.mul(other)

    def __truediv__(self, other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]) -> Self:
        if not isinstance(other, (self.__class__, Fits, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.div(other)

    def __rtruediv__(self, other: Union[Self, float, int, List[Union[Fits, float, int]]]) -> Self:
        if not isinstance(other, (self.__class__, float, int, List)):
            self.logger.error(f"Other must be either {self.__class__.__name__}, Fits, float or int")
            raise NotImplementedError

        return self.div(other).pow(-1)

    @classmethod
    def from_paths(cls, paths: List[str]) -> Self:
        """
        Create a `FitsArray` from paths as list of strings

        Parameters
        ----------
        paths : List[str]
            list of fits file paths

        Returns
        -------
        FitsArray
            the `FitsArray` created from list of fits files

        Raises
        ------
        NumberOfElementError
            when the number of fits files is 0
        """
        files = []
        for each in map(Path, paths):
            try:
                files.append(Fits(each))
            except FileNotFoundError:
                pass

        return cls(files)

    @classmethod
    def from_pattern(cls, pattern: str) -> Self:
        """
        Create a `FitsArray` from patterns

        Parameters
        ----------
        pattern : str
            the pattern that can be interpreted by glob

        Returns
        -------
        FitsArray
            the `FitsArray` created from pattern

        Raises
        ------
        NumberOfElementError
            when the number of fits files is 0
        """
        return cls.from_paths(glob(pattern))

    @classmethod
    def sample(cls, numer_of_samples: int = 10) -> Self:
        """
        Creates a sample `FitsArray` object
        see: http://www.astropy.org/astropy-data/tutorials/FITS-images/HorseHead.fits

        Parameters
        ----------
        numer_of_samples : int, default=10
            number of `Fits` in `FitsArray`

        Returns
        -------
        Self
            a `FitsArray` object.
        """
        fits_objects = []
        for i in range(numer_of_samples):
            try:
                f = Fits.sample()
                shifted = f.shift(i * 10, i * 10)
                fits_objects.append(shifted)
            except Exception:
                pass
        return cls(fits_objects)

    def merge(self, other: Self) -> None:
        """
        Merges two `FitsArray`s to create another `FitsArray`

        Parameters
        ----------
        other : FitsArray
            the other `FitsArray` to append to this one
        """
        self.logger.info("Merging")

        if not isinstance(other, FitsArray):
            self.logger.error(f"Other must be a {self.__class__.__name__}")
            raise ValueError(f"Other must be a {self.__class__.__name__}")

        self.fits_list.extend(other.fits_list)

    def append(self, other: Fits) -> None:
        """
        Appends a `Fits` to a `FitsArray`

        Parameters
        ----------
        other : Fits
            the other `Fits` to append to `FitsArray`
        """
        self.logger.info("Appending")

        if not isinstance(other, Fits):
            self.logger.error(f"Other must be a {self.__class__.__name__}")
            raise ValueError(f"Other must be a {self.__class__.__name__}")

        self.fits_list.append(other)

    def header(self) -> pd.DataFrame:
        """
        Returns headers of the fits files

        Returns
        -------
        pd.DataFrame
            the headers as dataframe
        """
        self.logger.info("Getting header")

        return pd.concat(
            (
                each.header()
                for each in self
            )
        )

    def data(self) -> List[Any]:
        """
        returns the data of fits files

        Returns
        -------
        List[Any]
            the list of data as `np.ndarray`
        """
        self.logger.info("Getting data")

        data = []

        for each in self:
            try:
                data.append(each.data())
            except ValueError:
                pass

        return data

    def value(self, x: int, y: int) -> pd.DataFrame:
        """
        Returns a table of values of asked coordinate

        Parameters
        ----------
        x : int
            x coordinate of asked pixel
        y: int
            y coordinate of asked pixel
        Returns
        -------
        pd.DataFrame
            table of values of asked coordinate
        """

        data = []
        for fits in self:
            try:
                value = fits.value(x, y)
                data.append([fits.file.name, value])
            except IndexError:
                pass

        return pd.DataFrame(
            data, columns=["image", "value"]
        ).set_index("image")

    def pure_header(self) -> List[Header]:
        """
        Returns the `Header` of the files

        Returns
        -------
        Header
            the list of Header object of the files
        """
        self.logger.info("Getting header (as an astropy header object)")

        data = []

        for each in self:
            data.append(each.pure_header())

        return data

    def ccd(self) -> List[CCDData]:
        """
        Returns the CCDData of the given files

        Returns
        -------
        CDDData
            the list of CCDData of the files
        """
        self.logger.info("Getting CCDData")

        data = []

        for each in self:
            data.append(each.ccd())

        return data

    def imstat(self) -> pd.DataFrame:
        """
        Returns statistics of the data

        Notes
        -----
        Stats are calculated using numpy and are:

        - number of pixels
        - mean
        - standard deviation
        - min
        - max

        Returns
        -------
        pd.DataFrame
            the statistics as dataframe
        """
        self.logger.info("Calculating image statistics")

        return pd.concat(
            [
                each.imstat()
                for each in self
            ]
        )

    def hedit(self, keys: Union[str, List[str]],
              values: Optional[Union[str, List[str]]] = None,
              delete: bool = False,
              value_is_key: bool = False) -> Self:
        """
        Edits header of the given files.

        Parameters
        ----------
        keys: str or List[str]
            Keys to be altered.
        values: str or List[str], optional
            Values to be added to set be set.
            Would be ignored if delete is True.
        delete: bool, optional
            Deletes the key from header if True.
        value_is_key: bool, optional
            Adds value of the key given in values if True. Would be ignored if
            delete is True.
        """
        self.logger.info("Editing header")

        for fits in self:
            try:
                fits.hedit(keys, values=values, delete=delete,
                           value_is_key=value_is_key)
            except Exception:
                pass

        return self

    def hselect(self, fields: Union[str, List[str]]) -> pd.DataFrame:
        """
        returns data frame containing wanted keys

        Parameters
        ----------
        fields: Union[str, List[str]]
            wanted fields

        Returns
        -------
        pd.DataFrame
            header values of give keys as data frame
        """
        self.logger.info("Returning selected headers")

        if isinstance(fields, str):
            fields = [fields]

        fields_to_use = []
        headers = self.header()

        for field in fields:
            if field in headers.columns:
                fields_to_use.append(field)

        if len(fields_to_use) < 1:
            return pd.DataFrame()

        return self.header()[fields_to_use]

    def save_as(self, output: str) -> Self:
        """
        Saves the `FitsArray` to output.

        Parameters
        ----------
        output: str
            New path to save the file.

        Returns
        -------
        Self
            New `FitsArray` object of saved `FitsArray`.

        Raises
        ------
        NumberOfElementError
            when the number of fits files is 0
        """
        self.logger.info("Saving all fits to as")

        output_fits = Fixer.outputs(output, self)
        fits_array = []
        for fits, output_fit in zip(self, output_fits):

            if output_fit is None:
                continue

            copied = fits.save_as(output_fit)
            fits_array.append(copied)

        return self.__class__(fits_array)

    def __prepare_weights(self,
                          weights: Optional[Union[List[str], List[Union[float, int]]]] = None
                          ) -> Optional[List[Union[int, float]]]:
        if weights is None:
            return None

        if len(weights) != len(self):
            raise NumberOfElementError("Number of Fits must be equal to number of weights")

        weights_to_use = []
        for fits, weight in zip(self, weights):
            if isinstance(weight, str):
                header = fits.header()
                if weight in header:
                    weights_to_use.append(header[weight].iloc[0])
                else:
                    raise ValueError("Header not available")

            elif isinstance(weight, (float, int)):
                weights_to_use.append(weight)
            else:
                raise ValueError("Weight must be either a header key or numeric value")

        return weights_to_use

    def __prepare_arith(self,
                        other: Union[
                            FitsArray, Fits, float, int,
                            List[Union[Fits, float, int]]
                        ]
                        ) -> Union[FitsArray, list[Union[Fits, float, int]]]:
        """
        Prepare the other for arithmetic operations

        Parameters
        ----------
        other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]
            The other value of arithmetic operation

        Returns
        -------
        Union[FitsArray, list[Union[Fits, float, int]]]
            the other value

        Raises
        ------
        ValueError
            when other is not correct
        NumberOfElementError
            when the length of other is wrong
        """
        self.logger.info("Preparing for arithmetic operations")

        other_to_use: Union[FitsArray, list[Union[Fits, float, int]]]

        if isinstance(other, (Fits, float, int)):
            other_to_use = [other] * len(self)
        elif isinstance(other, (FitsArray, List)):
            if len(other) != len(self):
                raise NumberOfElementError("Other must have the same length "
                                           "with the FitsArray")
            other_to_use = other
        else:
            raise ValueError("other must be either a value or list of values")

        return other_to_use

    def add(self,
            other: Union[
                FitsArray, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        """
        Does Addition operation on the `FitsArray` object

        Notes
        -----
        It is able to add numeric values, other `Fits`, list of numeric value or `FitsArray`

        - If other is numeric each element of the matrix will be added to the number.
        - If other is another `Fits` elementwise summation will be done.
        - If other is list of numeric the first would be applied to each matrix. Number of elements in list of numerics and `FitsArray` must be equal
        - If other is another `FitsArray` the second would be applied to each matrix. Number of elements in the both `FitsArray`s must be equal

        Parameters
        ----------
        other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]
            Either a `FitsArray` object, list of floats, list of integers,
            `Fits` object, float, or integer
        output: str
            New path to save the files.

        Returns
        -------
        Fits
            New `FitsArray` object of saved fits files.

        Raises
        ------
        NumberOfElementError
            when the length of other is wrong
        """
        self.logger.info("Making addition operation")

        other_to_use = self.__prepare_arith(other)

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, the_other, output_fit in zip(self, other_to_use, outputs):
            try:
                result = fits.add(the_other, output_fit)
                fits_array.append(result)
            except Exception:
                pass

        return self.__class__(fits_array)

    def sub(self,
            other: Union[
                FitsArray, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        """
        Does Subtraction operation on the `FitsArray` object


        Notes
        -----
        It is able to add numeric values, other `Fits`, list of numeric value or `FitsArray`

        - If other is numeric each element of the matrix will be subtracted from the number.
        - If other is another `Fits` elementwise subtraction will be done.
        - If other is list of numeric the first would be applied to each matrix. Number of elements in list of numerics and `FitsArray` must be equal
        - If other is another `FitsArray` the second would be applied to each matrix. Number of elements in the both `FitsArray`s must be equal


        Parameters
        ----------
        other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]
            Either a `FitsArray` object, list of floats, list of integers,
            `Fits` object, float, or integer
        output: str
            New path to save the files.

        Returns
        -------
        Fits
            New `FitsArray` object of saved fits files.

        Raises
        ------
        NumberOfElementError
            when the length of other is wrong
        """
        self.logger.info("Making subtraction operation")

        other_to_use = self.__prepare_arith(other)

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, the_other, output_fit in zip(self, other_to_use, outputs):
            try:
                result = fits.sub(the_other, output_fit)
                fits_array.append(result)
            except Exception:
                pass

        return self.__class__(fits_array)

    def mul(self,
            other: Union[
                FitsArray, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        """
        Does Multiplication operation on the `FitsArray` object


        Notes
        -----
        It is able to add numeric values, other `Fits`, list of numeric value or `FitsArray`

        - If other is numeric each element of the matrix will be multiplied by the number.
        - If other is another `Fits` elementwise multiplication will be done.
        - If other is list of numeric the first would be applied to each matrix. Number of elements in list of numerics and `FitsArray` must be equal
        - If other is another `FitsArray` the second would be applied to each matrix. Number of elements in the both `FitsArray`s must be equal


        Parameters
        ----------
        other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]
            Either a `FitsArray` object, list of floats, list of integers,
            `Fits` object, float, or integer
        output: str
            New path to save the files.

        Returns
        -------
        Fits
            New `FitsArray` object of saved fits files.

        Raises
        ------
        NumberOfElementError
            when the length of other is wrong
        """
        self.logger.info("Making multiplication operation")

        other_to_use = self.__prepare_arith(other)

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, the_other, output_fit in zip(self, other_to_use, outputs):
            try:
                result = fits.mul(the_other, output_fit)
                fits_array.append(result)
            except Exception:
                pass

        return self.__class__(fits_array)

    def div(self,
            other: Union[
                FitsArray, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        """
        Does Division operation on the `FitsArray` object


        Notes
        -----
        It is able to add numeric values, other `Fits`, list of numeric value or `FitsArray`

        - If other is numeric each element of the matrix will be divided by the number.
        - If other is another `Fits` elementwise division will be done.
        - If other is list of numeric the first would be applied to each matrix. Number of elements in list of numerics and `FitsArray` must be equal
        - If other is another `FitsArray` the second would be applied to each matrix. Number of elements in the both `FitsArray`s must be equal


        Parameters
        ----------
        other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]
            Either a `FitsArray` object, list of floats, list of integers,
            `Fits` object, float, or integer
        output: str
            New path to save the files.

        Returns
        -------
        Fits
            New `FitsArray` object of saved fits files.

        Raises
        ------
        NumberOfElementError
            when the length of other is wrong
        """
        self.logger.info("Making division operation")

        other_to_use = self.__prepare_arith(other)

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, the_other, output_fit in zip(self, other_to_use, outputs):
            try:
                result = fits.div(the_other, output_fit)
                fits_array.append(result)
            except Exception:
                pass

        return self.__class__(fits_array)

    def pow(self,
            other: Union[
                Self, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        """
        Does Power operation on the `FitsArray` object


        Notes
        -----
        It is able to raise numeric values, other `Fits`, list of numeric value or `FitsArray`

        - If other is numeric each element of the matrix will be raised by the number.
        - If other is another `Fits` elementwise power will be done.
        - If other is list of numeric the first would be applied to each matrix. Number of elements in list of numerics and `FitsArray` must be equal
        - If other is another `FitsArray` the second would be applied to each matrix. Number of elements in the both `FitsArray`s must be equal


        Parameters
        ----------
        other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]
            Either a `FitsArray` object, list of floats, list of integers,
            `Fits` object, float, or integer
        output: str
            New path to save the files.

        Returns
        -------
        Fits
            New `FitsArray` object of saved fits files.

        Raises
        ------
        NumberOfElementError
            when the length of other is wrong
        """
        self.logger.info("Making power operation")

        other_to_use = self.__prepare_arith(other)

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, the_other, output_fit in zip(self, other_to_use, outputs):
            try:
                result = fits.pow(the_other, output_fit)
                fits_array.append(result)
            except Exception:
                pass

        return self.__class__(fits_array)

    def imarith(self, other: Union[
        FitsArray, Fits, float, int, List[Union[Fits, float, int]]
    ],
                operand: str, output: Optional[str] = None) -> Self:
        """
        Does Arithmetic operation on the `FitsArray` object


        Notes
        -----
        It is able to add numeric values, other `Fits`, list of numeric value or `FitsArray`

        - If other is numeric each element of the matrix will be processed by the number.
        - If other is another `Fits` elementwise operation will be done.
        - If other is list of numeric the first would be applied to each matrix. Number of elements in list of numerics and `FitsArray` must be equal
        - If other is another `FitsArray` the second would be applied to each matrix. Number of elements in the both `FitsArray`s must be equal


        Parameters
        ----------
        other: Union[Self, Fits, float, int, List[Union[Fits, float, int]]]
            Either a `FitsArray` object, list of floats, list of integers,
            `Fits` object, float, or integer
        operand: str
            operation as string. One of `["+", "-", "*", "/", "**", "^"]`
        output: str
            New path to save the files.

        Returns
        -------
        Fits
            New `FitsArray` object of saved fits files.

        Raises
        ------
        NumberOfElementError
            when the length of other is wrong
        """
        self.logger.info("Making an arithmetic operation")

        other_to_use = self.__prepare_arith(other)

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, the_other, output_fit in zip(self, other_to_use, outputs):
            try:
                result = fits.imarith(the_other, operand, output_fit)
                fits_array.append(result)
            except Exception:
                pass

        return self.__class__(fits_array)

    def shift(self, xs: Union[List[int], int], ys: Union[List[int], int],
              output: Optional[str] = None) -> Self:
        """
        Shifts the data of `FitsArray` object

        Parameters
        ----------
        xs: Union[List[int], int]
            x coordinate(s)
        ys: Union[List[int], int]
            y coordinate(s)
        output: str, optional
            New path to save the files.

        Returns
        -------
        Self
            shifted `FitsArray` object
        """
        self.logger.info("Shifting all images")

        if isinstance(xs, int):
            to_x_shift = [xs] * len(self)

        elif isinstance(xs, list):
            to_x_shift = xs
        else:
            raise ValueError("xs must be either int or a list of int")

        if isinstance(ys, int):
            to_y_shift = [ys] * len(self)
        elif isinstance(ys, list):
            to_y_shift = ys
        else:
            raise ValueError("ys must be either int or a list of int")

        if len(to_x_shift) != len(to_y_shift) != len(self):
            raise NumberOfElementError("Number of xs, ys, and Fits in "
                                       "FitsArray must be equal")

        fits_array = []
        outputs = Fixer.outputs(output, self)

        for fits, output_fit, x, y in zip(self, outputs, to_x_shift,
                                          to_y_shift):
            try:
                aligned = fits.shift(x, y, output_fit)
                fits_array.append(aligned)
            except Exception:
                pass

        return self.__class__(fits_array)

    def rotate(self, angle: Union[List[Union[float]], float, int], reshape: bool = False,
               output: Optional[str] = None) -> Self:
        """
        Rotates the data of `FitsArray` object

        Parameters
        ----------
        angle: Union[List[Union[float]], float, int]
            Rotation angle(s)
        reshape: bool, default=False
            Reshape after rotate
        output: str, optional
            New path to save the files.

        Returns
        -------
        Self
            rotated `FitsArray` object
        """
        if isinstance(angle, (float, int)):
            to_rotate = [angle] * len(self)

        elif isinstance(angle, list):
            to_rotate = angle
        else:
            raise ValueError("rotate must be either float or a list of float")

        if len(to_rotate) != len(self):
            raise NumberOfElementError("Number of rotate and Fits in "
                                       "FitsArray must be equal")

        fits_array = []
        outputs = Fixer.outputs(output, self)

        for fits, output_fit, ang in zip(self, outputs, to_rotate):

            try:
                rotated = fits.rotate(ang, reshape=reshape, output=output_fit)
                fits_array.append(rotated)
            except Exception:
                pass

        return self.__class__(fits_array)

    def align(self, reference: Union[Fits, int] = 0, output: Optional[str] = None,
              max_control_points: int = 50, min_area: int = 5) -> Self:
        """
        Aligns the fits files with the given reference

        [1]: https://astroalign.quatrope.org/en/latest/api.html#astroalign.register

        Parameters
        ----------
        reference: Optional[Union[Fits, int]], default=0
            The reference Image or the index of `Fits` object in the `FitsArray`
             to be aligned as a Fits object.
        output: str, optional
            New path to save the files.
        max_control_points: int, default=50
            The maximum number of control point-sources to
            find the transformation. [1]
        min_area: int, default=5
            Minimum number of connected pixels to be considered a source. [1]

        Returns
        -------
        Self
            `FitsArray` object of aligned images.
        """
        self.logger.info("Aligning all images")

        if isinstance(reference, int):
            the_reference = self[reference]
        elif isinstance(reference, Fits):
            the_reference = reference
        else:
            self.logger.error("other must be either an integer or a Fits")
            raise ValueError("other must be either an integer or a Fits")

        if isinstance(the_reference, FitsArray):
            self.logger.error("Cannot be FitsArray")
            raise ValueError("Cannot be FitsArray")

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, output_fit in zip(self, outputs):
            try:
                aligned = fits.align(the_reference, output_fit,
                                     max_control_points=max_control_points,
                                     min_area=min_area
                                     )
                fits_array.append(aligned)
            except Exception:
                pass

        return self.__class__(fits_array)

    def solve_filed(self) -> Self:
        """
        Creates the WCS headers
        """
        self.logger.info("Solving field")

        return self

    def zero_correction(self, master_zero: Fits, output: Optional[str] = None,
                        force: bool = False) -> Self:
        """
        Does zero correction of the data

        Parameters
        ----------
        master_zero : Fits
            Zero file to be used for correction
        output: str, optional
            New path to save the files.
        force: bool, default=False
            Overcorrection flag

        Returns
        -------
        Self
            Zero corrected `FitsArray` object
        """
        self.logger.info("Making zero correction on the image")

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, output_fit in zip(self, outputs):
            try:
                zero_corrected = fits.zero_correction(master_zero,
                                                      output=output_fit,
                                                      force=force)
                fits_array.append(zero_corrected)
            except OverCorrection:
                fits_array.append(fits)
            except Exception:
                pass

        return self.__class__(fits_array)

    def dark_correction(self, master_dark: Fits,
                        exposure: Optional[str] = None,
                        output: Optional[str] = None,
                        force: bool = False) -> Self:
        """
        Does dark correction of the data

        Parameters
        ----------
        master_dark : Fits
            Dark file to be used for correction
        exposure : str, optional
            header card containing exptime
        output: str, optional
            New path to save the files.
        force: bool, default=False
            Overcorrection flag

        Returns
        -------
        Self
            Dark corrected `FitsArray` object

        """
        self.logger.info("Making dark correction on the image")

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, output_fit in zip(self, outputs):
            try:
                dark_corrected = fits.dark_correction(master_dark,
                                                      exposure=exposure,
                                                      output=output_fit,
                                                      force=force)
                fits_array.append(dark_corrected)
            except OverCorrection:
                fits_array.append(fits)
            except Exception:
                pass

        return self.__class__(fits_array)

    def flat_correction(self, master_flat: Fits, output: Optional[str] = None,
                        force: bool = False) -> Self:
        """
        Does flat correction of the data

        Parameters
        ----------
        master_flat : Fits
            Flat file to be used for correction
        output: str, optional
            New path to save the files.
        force: bool, default=False
            Overcorrection flag

        Returns
        -------
        Self
            Flat corrected `FitsArray` object
        """
        self.logger.info("Making flat correction on the image")

        fits_array = []
        outputs = Fixer.outputs(output, self)
        for fits, output_fit in zip(self, outputs):
            try:
                flat_corrected = fits.flat_correction(master_flat,
                                                      output=output_fit,
                                                      force=force)
                fits_array.append(flat_corrected)
            except OverCorrection:
                fits_array.append(fits)
            except Exception:
                pass

        return self.__class__(fits_array)

    def background(self) -> List[Background]:
        """
        Returns a list of `Background` objects of the fits files.
        """
        self.logger.info("Getting background")

        data = []

        for fits in self:
            data.append(fits.background())

        return data

    def daofind(self, index: int = 0, sigma: float = 3.0, fwhm: float = 3.0,
                threshold: float = 5.0) -> pd.DataFrame:
        """
        Runs daofind to detect sources on the image.

        [1]: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html

        [2]: https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html

        Parameters
        ----------
        index: int, default=0
            The index of `Fits` in `FitsArray` to run daofind on
        sigma: float, default=3
            The number of standard deviations to use for both the lower and
            upper clipping limit. These limits are overridden by sigma_lower
            and sigma_upper, if input.
            The default is 3. [1]
        fwhm: float, default=3
            The full-width half-maximum (FWHM) of the major axis of the
            Gaussian kernel in units of pixels. [2]
        threshold: float, default=5
            The absolute image value above which to select sources. [2]

        Returns
        -------
        pd.DataFrame
            List of sources found on the image.
        """
        self.logger.info("Extracting sources (daofind) from images")

        return self[index].daofind(sigma=sigma, fwhm=fwhm, threshold=threshold)

    def extract(self, index: int = 0, detection_sigma: float = 5.0,
                min_area: float = 5.0) -> pd.DataFrame:
        """
        Runs astroalign._find_sources to detect sources on the image.

        Parameters
        ----------
        index: int, default=0
            The index of `Fits` in `FitsArray` to run extract on
        detection_sigma: float, default=5
            `thresh = detection_sigma * bkg.globalrms`
        min_area: float, default=5
            Minimum area

        Returns
        -------
        pd.DataFrame
            List of sources found on the image.
        """
        self.logger.info("Extracting sources (sep_extract) from images")

        return self[index].extract(
            detection_sigma=detection_sigma, min_area=min_area
        )

    def photometry_sep(self, xs: NUMERICS, ys: NUMERICS, rs: NUMERICS,
                       headers: Optional[Union[str, list[str]]] = None,
                       exposure: Optional[Union[str, float, int]] = None
                       ) -> pd.DataFrame:
        """
        Does a photometry using sep

        Parameters
        ----------
        xs: Union[float, int, List[Union[float, int]]]
            x coordinate(s)
        ys: Union[float, int, List[Union[float, int]]]
            y coordinate(s)
        rs: Union[float, int, List[Union[float, int]]]
            aperture(s)
        headers: Union[str, list[str]], optional
            Header keys to be extracted after photometry
        exposure: Union[str, float, int], optional
            Header key that contains or a numeric value of exposure time

        Returns
        -------
        pd.DataFrame
            photometric data as dataframe

        Raises
        ------
        NumberOfElementError
            when `x` and `y` coordinates does not have the same length
        """
        self.logger.info("Doing photometry (sep) on the image")

        photometry = []
        for fits in self:
            try:
                phot = fits.photometry_sep(xs, ys, rs, headers=headers,
                                           exposure=exposure)
                photometry.append(phot)
            except NumberOfElementError:
                raise NumberOfElementError(
                    "The length of Xs and Ys must be equal"
                )
            except Exception:
                pass

        if len(photometry) < 1:
            return pd.DataFrame()

        return pd.concat(photometry)

    def photometry_phu(self, xs: NUMERICS, ys: NUMERICS, rs: NUMERICS,
                       headers: Optional[Union[str, list[str]]] = None,
                       exposure: Optional[Union[str, float, int]] = None
                       ) -> pd.DataFrame:
        """
        Does a photometry using photutils

        Parameters
        ----------
        xs: Union[float, int, List[Union[float, int]]]
            x coordinate(s)
        ys: Union[float, int, List[Union[float, int]]]
            y coordinate(s)
        rs: Union[float, int, List[Union[float, int]]]
            aperture(s)
        headers: Union[str, list[str]], optional
            Header keys to be extracted after photometry
        exposure: Union[str, float, int], optional
            Header key that contains or a numeric value of exposure time

        Returns
        -------
        pd.DataFrame
            photometric data as dataframe

        Raises
        ------
        NumberOfElementError
            when `x` and `y` coordinates does not have the same length
        """
        self.logger.info("Doing photometry (photutils) on the image")

        photometry = []
        for fits in self:
            try:
                phot = fits.photometry_phu(xs, ys, rs, headers=headers,
                                           exposure=exposure)
                photometry.append(phot)
            except NumberOfElementError:
                raise NumberOfElementError(
                    "The length of Xs and Ys must be equal"
                )
            except Exception:
                pass

        if len(photometry) < 1:
            return pd.DataFrame()

        return pd.concat(photometry)

    def photometry(self, xs: NUMERICS, ys: NUMERICS, rs: NUMERICS,
                   headers: Optional[Union[str, list[str]]] = None,
                   exposure: Optional[Union[str, float, int]] = None
                   ) -> pd.DataFrame:
        """
        Does a photometry using both sep and photutils

        Parameters
        ----------
        xs: Union[float, int, List[Union[float, int]]]
            x coordinate(s)
        ys: Union[float, int, List[Union[float, int]]]
            y coordinate(s)
        rs: Union[float, int, List[Union[float, int]]]
            aperture(s)
        headers: Union[str, list[str]], optional
            Header keys to be extracted after photometry
        exposure: Union[str, float, int], optional
            Header key that contains or a numeric value of exposure time

        Returns
        -------
        pd.DataFrame
            photometric data as dataframe

        Raises
        ------
        NumberOfElementError
            when `x` and `y` coordinates does not have the same length
        """
        photometry = []
        for fits in self:
            try:
                phot = fits.photometry(xs, ys, rs, headers=headers,
                                       exposure=exposure)
                photometry.append(phot)
            except Exception:
                pass

        if len(photometry) < 1:
            return pd.DataFrame()

        return pd.concat(photometry)

    def cosmic_clean(self, output: Optional[str] = None,
                     override: bool = False, sigclip: float = 4.5,
                     sigfrac: float = 0.3, objlim: int = 5, gain: float = 1.0,
                     readnoise: float = 6.5, satlevel: float = 65535.0,
                     niter: int = 4, sepmed: bool = True,
                     cleantype: str = 'meanmask', fsmode: str = 'median',
                     psfmodel: str = 'gauss', psffwhm: float = 2.5,
                     psfsize: int = 7, psfk: Any = Optional[Any],
                     psfbeta: float = 4.765, gain_apply: bool = True) -> Self:
        """
        Clears cosmic rays from the fits files

        [1]: https://ccdproc.readthedocs.io/en/latest/api/ccdproc.cosmicray_lacosmic.html

        Parameters
        ----------
        output: str, optional
            Path of the new fits file.
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.
        sigclip: float, default=4.5
            Laplacian-to-noise limit for cosmic ray detection.
            Lower values will flag more pixels as cosmic rays.
            Default: 4.5. see [1]
        sigfrac: float, default=0.3
            Fractional detection limit for neighboring pixels.
            For cosmic ray neighbor pixels, a Laplacian-to-noise
            detection limit of sigfrac * sigclip will be used.
            Default: 0.3. see [1]
        objlim: int, default=5
            Minimum contrast between Laplacian image
            and the fine structure image.
            Increase this value if cores of bright stars are
            flagged as cosmic rays.
            Default: 5.0. see [1]
        gain: float, default=1.5
            Gain of the image (electrons / ADU).
            We always need to work in electrons for cosmic ray detection.
            Default: 1.0 see [1]
        readnoise: float, default=6.5
            Read noise of the image (electrons).
            Used to generate the noise model of the image.
            Default: 6.5. see [1]
        satlevel: float, default=65535.0
            Saturation level of the image (electrons).
            This value is used to detect saturated stars and pixels at or
            above this level are added to the mask.
            Default: 65535.0. see [1]
        niter: int, default=4
            Number of iterations of the LA Cosmic algorithm to perform.
            Default: 4. see [1]
        sepmed: bool, default=True
            Use the separable median filter instead of the full median filter.
            The separable median is not identical to the full median filter,
            but they are approximately the same,
            the separable median filter is significantly faster,
            and still detects cosmic rays well.
            Note, this is a performance feature,
            and not part of the original L.A. Cosmic.
            Default: True. see [1]
        cleantype: str, default='meanmask'
            Set which clean algorithm is used:
            1) "median": An unmasked 5x5 median filter.
            2) "medmask": A masked 5x5 median filter.
            3) "meanmask": A masked 5x5 mean filter.
            4) "idw": A masked 5x5 inverse distance weighted interpolation.
            Default: "meanmask". see [1]
        fsmode: float, default='median'
            Method to build the fine structure image:
            1) "median": Use the median filter in the standard LA
            Cosmic algorithm.
            2) "convolve": Convolve the image with the psf kernel to
            calculate the fine structure image.
            Default: "median". see [1]
        psfmodel: str, default='gauss'
            Model to use to generate the psf kernel if fsmode == ‘convolve’
            and psfk is None.
            The current choices are Gaussian and Moffat profiles:
            - "gauss" and "moffat" produce circular PSF kernels.
            - The "gaussx" and "gaussy" produce Gaussian kernels in the x
            and y directions respectively.
            Default: "gauss". see [1]
        psffwhm: float, default=2.5
            Full Width Half Maximum of the PSF to use to generate the kernel.
            Default: 2.5. see [1]
        psfsize: int, default=7
            Size of the kernel to calculate.
            Returned kernel will have size psfsize x psfsize.
            psfsize should be odd.
            Default: 7. see [1]
        psfk: Any, optional
            PSF kernel array to use for the fine structure image
            if fsmode == 'convolve'. If None and fsmode == 'convolve',
            we calculate the psf kernel using psfmodel.
            Default: None. see [1]
        psfbeta: float, default=4.765
            Moffat beta parameter. Only used if fsmode=='convolve' and
            psfmodel=='moffat'.
            Default: 4.765.
        gain_apply: bool, default=True
            If True, return gain-corrected data, with correct units,
            otherwise do not gain-correct the data.
            Default is True to preserve backwards compatibility. see [1]

        Returns
        -------
        Self
            Cleaned fits files
        """
        self.logger.info("Cleaning the data")

        outputs = Fixer.outputs(output, self)
        clean_fits_array = []
        for fits, out_fit in zip(self, outputs):
            try:
                clean_fits = fits.cosmic_clean(
                    out_fit,
                    override=override, sigclip=sigclip, sigfrac=sigfrac,
                    objlim=objlim, gain=gain, readnoise=readnoise,
                    satlevel=satlevel, niter=niter, sepmed=sepmed,
                    cleantype=cleantype, fsmode=fsmode, psfmodel=psfmodel,
                    psffwhm=psffwhm, psfsize=psfsize, psfk=psfk,
                    psfbeta=psfbeta,
                    gain_apply=gain_apply
                )
                clean_fits_array.append(clean_fits)
            except Exception:
                pass

        return self.__class__(clean_fits_array)

    def show(self, scale: bool = True, interval: float = 1.0) -> None:
        """
        Shows the Images using matplotlib.

        Parameters
        ----------
        scale: bool, optional
            Scales the Image if True.
        interval: float, default=1
            The interval of the animation
        """
        self.logger.info("Showing all images")

        fig = plt.figure()

        if scale:
            zscale = ZScaleInterval()
        else:
            def zscale(x):
                return x

        im = plt.imshow(zscale(self[0].data()), cmap="Greys_r", animated=True)
        plt.xticks([])
        plt.yticks([])

        def updatefig(args):
            im.set_array(zscale(self[args % len(self)].data()))
            return im,

        _ = animation.FuncAnimation(
            fig, updatefig, interval=interval, blit=True
        )
        plt.show()

    def group_by(self,
                 groups: Union[str, List[str]]
                 ) -> Dict[Hashable, Self]:
        """
        Groups the `FitsArray` by given header

        Parameters
        ----------
        groups: Union[str, List[str]]
            header keys

        Returns
        -------
        Dict[Hashable, FitsArray]
            header keys and `FitsArray` pairs
        """
        self.logger.info("Group images by headers")

        if isinstance(groups, str):
            groups = [groups]

        if len(groups) < 1:
            return dict()

        headers = self.header()

        for group in groups:
            if group not in headers.columns:
                headers[group] = "N/A"

        grouped = {}
        for keys, df in headers.fillna("N/A").groupby(groups, dropna=False):
            grouped[keys] = self.__class__.from_paths(df.index.tolist())

        return grouped

    def combine(self, method: str = "average", clipping: Optional[str] = None,
                weights: Optional[List[Union[float, int]]] = None) -> Fits:
        """
        Combines FitsArray to a Fits

        Parameters
        ----------
        method : str
            method of combine. Either average, mean, or median
        clipping: str, optional
            clipping method (same as rejection in IRAF). Either sigmaclip or minmax
        weights: float or int, optional
            weights to be applied before combining. If None [1, ...] will be used.

        Returns
        -------
        Fits
            the combined `Fits`

        Raises
        ------
        ValueError
            when the number weight is not equal to number of fits files
        ValueError
            when the method is not either of average, mean, median, or sum
        ValueError
            when the clipping is not either of sigma, or minmax
        """
        Check.method(method)
        Check.clipping(clipping)

        combiner = Combiner(self.ccd())

        if weights is None:
            weights = [1] * len(self)

        if len(weights) != len(self):
            raise ValueError("Length of weights must be equal to number of Fits")

        if clipping is not None:
            if "sigma".startswith(clipping):
                combiner.sigma_clipping()
            else:
                combiner.minmax_clipping()

        combiner.weights = np.array(weights)

        if "median".startswith(method.lower()):
            return Fits.from_data_header(data=combiner.median_combine().data)
        elif "sum".startswith(method.lower()):
            return Fits.from_data_header(data=combiner.sum_combine().data)
        else:
            return Fits.from_data_header(data=combiner.average_combine().data)

    def zero_combine(self, method: str = "median", clipping: Optional[str] = None) -> Fits:
        """
        Combines FitsArray to a Fits optimized for zero combining

        Parameters
        ----------
        method : str
            method of combine. Either average, mean, or median
        clipping: str, optional
            clipping method (same as rejection in IRAF). Either sigmaclip or minmax

        Returns
        -------
        Fits
            the combined `Fits`

        Raises
        ------
        ValueError
            when the method is not either of average, mean, median, or sum
        ValueError
            when the clipping is not either of sigma, or minmax
        """
        return self.combine(method=method, clipping=clipping)

    def dark_combine(self, method: str = "median", clipping: Optional[str] = None,
                     weights: Optional[Union[List[str], List[Union[float, int]]]] = None) -> Fits:
        """
        Combines FitsArray to a Fits optimized for dark combining

        Parameters
        ----------
        method : str
            method of combine. Either average, mean, or median
        clipping: str, optional
            clipping method (same as rejection in IRAF). Either sigmaclip or minmax
        weights: str, float, or int, optional
            weights to be applied before combining. If None [1, ...] will be used.

        Returns
        -------
        Fits
            the combined `Fits`

        Raises
        ------
        ValueError
            when the number weight is not equal to number of fits files
        ValueError
            when the method is not either of average, mean, median, or sum
        ValueError
            when the clipping is not either of sigma, or minmax
        """

        fixed_weights = self.__prepare_weights(weights)
        return self.combine(method=method, clipping=clipping, weights=fixed_weights)

    def flat_combine(self, method: str = "median", clipping: Optional[str] = None,
                     weights: Optional[Union[List[str], List[Union[float, int]]]] = None) -> Fits:
        """
        Combines FitsArray to a Fits optimized for flat combining

        Parameters
        ----------
        method : str
            method of combine. Either average, mean, or median
        clipping: str, optional
            clipping method (same as rejection in IRAF). Either sigmaclip or minmax
        weights: str, float, or int, optional
            weights to be applied before combining. If None [1, ...] will be used.

        Returns
        -------
        Fits
            the combined `Fits`

        Raises
        ------
        ValueError
            when the number weight is not equal to number of fits files
        ValueError
            when the method is not either of average, mean, median, or sum
        ValueError
            when the clipping is not either of sigma, or minmax
        """
        fixed_weights = self.__prepare_weights(weights)
        return self.combine(method=method, clipping=clipping, weights=fixed_weights)
