from __future__ import annotations

from .error import NothingToDo, AlignError, NumberOfElementError, \
    OverCorrection, ValueNotFound
from .models import Data, NUMERICS
from .utils import Fixer, Check

import math
import shutil
from typing import Optional, Union, List, Any, Tuple

from photutils.aperture import CircularAperture, aperture_photometry
from photutils.utils import calc_total_error
from typing_extensions import Self

import astroalign
from astropy import units
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder

import numpy as np
import pandas as pd
from pathlib import Path

from astropy.io.fits.header import Header
from astropy.io import fits as fts

from sep import extract as sep_extract, Background, sum_circle

from ccdproc import cosmicray_lacosmic, subtract_bias, subtract_dark, \
    flat_correct

from matplotlib import pyplot as plt
from mpl_point_clicker import clicker


class Fits(Data):
    def __init__(self, file: Path) -> None:

        self.is_temp = False
        if not file.exists():
            raise FileNotFoundError("File does not exist")
        self.file = file
        self.ZMag = 25

    def __str__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"(@: '{id(self)}', path:'{self.file}')"

    def __repr__(self):
        return f"{self.__class__.__name__}.from_path('{self.file}')"

    def __del__(self):
        if self.is_temp:
            self.file.unlink()

    def __abs__(self) -> str:
        return str(self.file.absolute())

    def flux_to_mag(self, flux: Union[int, float],
                    flux_error: Union[int, float],
                    exptime: Union[int, float]
                    ) -> Tuple[Union[int, float], Union[int, float]]:
        r"""
        Converts flux and flux error to magnitude and magnitude error

        Notes
        -----
        We use an approximation to calculate mag and merr
        see: https://github.com/spacetelescope/wfc3_photometry/blob/71a40892d665118d161da27465474778b4cf9f1f/photometry_tools/photometry_with_errors.py#L127

        .. math::
            mag = -2.5 * log(f) + 2.5 * log(t_e)

            m_{err} = 1.0857 \times \frac{f}{f_e}

        Where :math:`f` is flux, :math:`t_e` is exposure time, and :math:`f_e` is flux error.

        Parameters
        ----------
        flux : Union[int, float]
            measured flux
        flux_error : Union[int, float]
            measured flux error
        exptime : Union[int, float]
            exposure time

        Returns
        -------
        Tuple[Union[int, float], Union[int, float]]
            calculated magnitude and magnitude error.

        Raises
        ------
        ZeroDivisionError
            when the `flux` is `0`
        """
        mag = -2.5 * math.log10(flux)
        if exptime != 0:
            mag += 2.5 * math.log10(exptime)

        if flux_error <= 0:
            mag_err = 0.0
        else:
            mag_err = 1.0857 * flux_error / flux

        if math.isinf(mag_err):
            mag_err = 0

        return mag + self.ZMag, mag_err

    @classmethod
    def from_path(cls, path: str) -> Self:
        """
        Creates a `Fits` object from the given file `path` as string

        Parameters
        ----------
        path : str
            path of the file as string

        Returns
        -------
        Self
            a `Fits` object.

        Raises
        ------
        FileNotFoundError
            when the file does not exist
        """
        return cls(Path(path))

    @classmethod
    def from_data_header(cls, data: Any,
                         header: Optional[Header] = None,
                         output: Optional[str] = None,
                         override: bool = False) -> Self:
        """
        Creates a `Fits` object th give `data` and `header`

        Parameters
        ----------
        data : Any
            the data as `np.ndarray`
        header : Header
            the header as `Header`
        output : str, optional
            the wanted file path.
            a temporary file will be created if it's `None`
        override : bool, default=False
            delete already existing file if `true`

        Returns
        -------
        Self
            a `Fits` object.

        Raises
        ------
        FileExistsError
            when the file does exist and `override` is `False`
        """
        new_output = Fixer.output(output=output, override=override)
        fts.writeto(new_output, data, header=header)
        fits = cls.from_path(new_output)
        fits.is_temp = output is None
        return fits

    @classmethod
    def sample(cls) -> Self:
        """
        Creates a sample `Fits` object
        see: http://www.astropy.org/astropy-data/tutorials/FITS-images/HorseHead.fits


        Returns
        -------
        Self
            a `Fits` object.
        """
        file = str(Path(__file__).parent / "sample.fits")
        data = fts.getdata(file)
        header = fts.getheader(file)
        return cls.from_data_header(data, header=header)

    def reset_zmag(self):
        """
        Resets Zmag value to 25

        Notes
        -----
        ZMag is the value added to calculated magnitude from flux.

        .. math::
            mag = ZMag + mag_c

        Where :math:`ZMag` is Zero Magnitude, :math:`mag_c` is calculated magnitude
        """
        self.ZMag = 25

    def header(self) -> pd.DataFrame:
        """
        Returns headers of the fits file

        Returns
        -------
        pd.DataFrame
            the headers as dataframe
        """
        header = fts.getheader(abs(self))
        return pd.DataFrame(
            {i: header[i] for i in header if i}, index=[0]).assign(
            image=[self.file.name]
        ).set_index("image")

    def data(self) -> Any:
        """
        returns the data of fits file

        Returns
        -------
        Any
            the data as `np.ndarray`

        Raises
        ------
        ValueError
            if the fits file is not an image
        """
        data = fts.getdata(abs(self))
        if not isinstance(data, np.ndarray):
            raise ValueError("Unknown Fits type")

        return data.astype(float)

    def pure_header(self) -> Header:
        """
        Returns the `Header` of the file

        Returns
        -------
        Header
            the Header object of the file
        """
        return fts.getheader(abs(self))

    def ccd(self) -> CCDData:
        """
        Returns the CCDData of the given file

        Returns
        -------
        CDDData
            the CCDData of the file
        """
        return CCDData.read(self.file, unit="adu")

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
        data = self.data()
        return pd.DataFrame(
            [
                [
                    self.file.name, data.size, np.mean(data), np.std(data),
                    np.min(data), np.max(data)
                ]
            ],
            columns=["image", "npix", "mean", "stddev", "min", "max"]
        ).set_index("image")

    def cosmic_clean(self, output: Optional[str] = None,
                     override: bool = False, sigclip: float = 4.5,
                     sigfrac: float = 0.3, objlim: int = 5, gain: float = 1.0,
                     readnoise: float = 6.5, satlevel: float = 65535.0,
                     niter: int = 4, sepmed: bool = True,
                     cleantype: str = 'meanmask', fsmode: str = 'median',
                     psfmodel: str = 'gauss', psffwhm: float = 2.5,
                     psfsize: int = 7, psfk: Any = None,
                     psfbeta: float = 4.765, gain_apply: bool = True) -> Self:
        """
        Clears cosmic rays from the fits file

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
            Cleaned fits
        """

        cleaned_data, _ = cosmicray_lacosmic(
            self.data(), sigclip=sigclip,
            sigfrac=sigfrac, objlim=objlim,
            gain=gain,
            readnoise=readnoise, satlevel=satlevel,
            niter=niter, sepmed=sepmed,
            cleantype=cleantype, fsmode=fsmode,
            psfmodel=psfmodel, psffwhm=psffwhm,
            psfsize=psfsize, psfk=psfk,
            psfbeta=psfbeta, gain_apply=gain_apply
        )

        return self.from_data_header(cleaned_data.value,
                                     header=self.pure_header(), output=output,
                                     override=override)

    def hedit(self, keys: Union[str, List[str]],
              values: Optional[Union[str, List[str]]] = None,
              delete: Optional[bool] = False,
              value_is_key: bool = False) -> Self:
        """
        Edits header of the given file.

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

        Returns
        -------
        Self
            The `Fits` object
        """
        if delete:
            if isinstance(keys, str):
                keys = [keys]

            with fts.open(abs(self), "update") as hdu:
                for key in keys:
                    if key in hdu[0].header:
                        del hdu[0].header[key]

        else:
            if values is None:
                raise NothingToDo("Delete is False and Value is not given")

            if not isinstance(values, type(keys)):
                raise ValueError(
                    "keys and values must have the same type (str or list)")

            if isinstance(keys, str):
                keys = [keys]

            if isinstance(values, str):
                values = [values]

            if len(keys) != len(values):
                raise ValueError(
                    "List of keys and values must be equal in length")

            with fts.open(abs(self), "update") as hdu:
                for key, value in zip(keys, values):
                    if value_is_key:
                        hdu[0].header[key] = hdu[0].header[value]
                    else:
                        hdu[0].header[key] = value

        return self

    def save_as(self, output: str, override: bool = False) -> Self:
        """
        Saves the `Fits` file as output.

        Parameters
        ----------
        output: str
            New path to save the file.
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.

        Returns
        -------
        Self
            New `Fits` object of saved fits file.

        Raises
        ------
        FileExistsError
            when the file does exist and `override` is `False`
        """
        new_output = Fixer.output(output=output, override=override)
        shutil.copy(self.file, new_output)
        return self.__class__.from_path(new_output)

    def add(self, other: Union[Fits, float, int], output: Optional[str] = None,
            override: bool = False) -> Self:
        r"""
        Does Addition operation on the `Fits` object

        Notes
        -----
        It is able to add numeric values as other `Fits`

        - If other is numeric each element of the matrix will be added to the number.
        - If other is another `Fits` elementwise summation will be done.

        Parameters
        ----------
        other: Union[Fits, float, int]
            Either a `Fits` object, float, or integer
        output: str
            New path to save the file.
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.

        Returns
        -------
        Self
            New `Fits` object of saved fits file.

        Raises
        ------
        FileExistsError
            when the file does exist and `override` is `False`
        """
        if not isinstance(other, (float, int, self.__class__)):
            raise ValueError(
                f"Please provide either a {self.__class__} "
                "Object or a numeric value"
            )

        if isinstance(other, (float, int)):
            new_data = self.data() + other
        else:
            new_data = self.data() + other.data()

        return self.__class__.from_data_header(
            new_data, header=self.pure_header(),
            output=output, override=override
        )

    def sub(self, other: Union[Fits, float, int], output: Optional[str] = None,
            override: bool = False) -> Self:
        """
        Does Subtraction operation on the `Fits` object

        Notes
        -----
        It is able to subtract numeric values as other `Fits`

        - If other is numeric each element of the matrix will be subtracted by the number.
        - If other is another `Fits` elementwise subtraction will be done.


        Parameters
        ----------
        other: Union[Fits, float, int]
            Either a `Fits` object, float, or integer
        output: str
            New path to save the file.
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.

        Returns
        -------
        Self
            New `Fits` object of saved fits file.

        Raises
        ------
        FileExistsError
            when the file does exist and `override` is `False`
        """
        if not isinstance(other, (float, int, self.__class__)):
            raise ValueError(
                f"Please provide either a {self.__class__} "
                "Object or a numeric value"
            )

        if isinstance(other, (float, int)):
            new_data = self.data() - other
        else:
            new_data = self.data() - other.data()

        return self.__class__.from_data_header(
            new_data, header=self.pure_header(),
            output=output, override=override
        )

    def mul(self, other: Union[Fits, float, int], output: Optional[str] = None,
            override: bool = False) -> Self:
        """
        Does Multiplication operation on the `Fits` object

        Notes
        -----
        It is able to multiply numeric values as other `Fits`

        - If other is numeric each element of the matrix will be multiplied by the number.
        - If other is another `Fits` elementwise multiplication will be done.


        Parameters
        ----------
        other: Union[Fits, float, int]
            Either a `Fits` object, float, or integer
        output: str
            New path to save the file.
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.

        Returns
        -------
        Self
            New `Fits` object of saved fits file.

        Raises
        ------
        FileExistsError
            when the file does exist and `override` is `False`
        """
        if not isinstance(other, (float, int, self.__class__)):
            raise ValueError(
                f"Please provide either a {self.__class__} "
                "Object or a numeric value"
            )

        if isinstance(other, (float, int)):
            new_data = self.data() * other
        else:
            new_data = self.data() * other.data()

        return self.__class__.from_data_header(
            new_data, header=self.pure_header(),
            output=output, override=override
        )

    def div(self, other: Union[Fits, float, int], output: Optional[str] = None,
            override: bool = False) -> Self:
        """
        Does Division operation on the `Fits` object

        Notes
        -----
        It is able to divide numeric values as other `Fits`

        - If other is numeric each element of the matrix will be divided by the number.
        - If other is another `Fits` elementwise division will be done.


        Parameters
        ----------
        other: Union[Fits, float, int]
            Either a `Fits` object, float, or integer
        output: str
            New path to save the file.
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.

        Returns
        -------
        Self
            New `Fits` object of saved fits file.

        Raises
        ------
        FileExistsError
            when the file does exist and `override` is `False`
        """
        if not isinstance(other, (float, int, self.__class__)):
            raise ValueError(
                f"Please provide either a {self.__class__} "
                "Object or a numeric value"
            )

        if isinstance(other, (float, int)):
            new_data = self.data() / other
        else:
            new_data = self.data() / other.data()

        return self.__class__.from_data_header(
            new_data, header=self.pure_header(),
            output=output, override=override
        )

    def imarith(self, other: Union[Fits, float, int],
                operand: str,
                output: Optional[str] = None,
                override: bool = False) -> Self:
        """
        Does Arithmetic operation on the `Fits` object

        Notes
        -----
        It is able to do operation with numeric values as other `Fits`

        - If other is numeric each element of the matrix will be processed by the number.
        - If other is another `Fits` elementwise operation will be done.


        Parameters
        ----------
        other: Union[Fits, float, int]
            Either a `Fits`, `float`, or `int`
        operand: str
            operation as string. One of `["+", "-", "*", "/"]`
        output: str
            New path to save the file.
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.

        Returns
        -------
        Self
            New `Fits` object of saved fits file.

        Raises
        ------
        ValueError
            when the given value is not `Fits`, `float`, or `int`
        ValueError
            when operand is not one of `["+", "-", "*", "/"]`
        """

        if not isinstance(other, (float, int, self.__class__)):
            raise ValueError(
                f"Please provide either a {self.__class__} "
                "Object or a numeric value"
            )

        Check.operand(operand)

        if operand == "+":
            return self.add(other, output=output, override=override)
        elif operand == "-":
            return self.sub(other, output=output, override=override)
        elif operand == "*":
            return self.mul(other, output=output, override=override)
        else:
            return self.div(other, output=output, override=override)

    def align(self, reference: Fits, output: Optional[str] = None,
              max_control_points: int = 50, min_area: int = 5,
              override: bool = False) -> Self:
        """
        Aligns the fits file with the given reference

        [1]: https://astroalign.quatrope.org/en/latest/api.html#astroalign.register

        Parameters
        ----------
        reference: Fits
            The reference Image to be aligned as a Fits object.
        output: str, optional
            Path of the new fits file.
        max_control_points: int, default=50
            The maximum number of control point-sources to
            find the transformation. [1]
        min_area: int, default=5
            Minimum number of connected pixels to be considered a source. [1]
        override: bool, default=False
            If True will overwrite the new_path if a file is already exists.

        Returns
        -------
        Self
            `Fits` object of aligned image.
        """

        if not isinstance(reference, self.__class__):
            raise ValueError(f"Other must be a {self.__class__}")

        try:
            registered_image, _ = astroalign.register(
                self.data(),
                reference.data(),
                max_control_points=max_control_points,
                min_area=min_area
            )

            return self.__class__.from_data_header(
                registered_image, header=fts.getheader(self.file),
                output=output, override=override
            )
        except ValueError:
            raise AlignError("Cannot align two images")

    def show(self, scale: bool = True,
             sources: Optional[pd.DataFrame] = None) -> None:
        """
        Shows the Image using matplotlib.

        Parameters
        ----------
        scale: bool, optional
            Scales the Image if True.
        sources: pd.DataFrame, optional
            Draws points on image if a list is given.
        """
        if scale:
            zscale = ZScaleInterval()
        else:
            def zscale(x):
                return x

        plt.imshow(zscale(self.data()), cmap="Greys_r")

        if sources is not None:
            plt.scatter(sources["xcentroid"], sources["ycentroid"])

        plt.xticks([])
        plt.yticks([])
        plt.show()

    def coordinate_picker(self, scale: bool = True) -> pd.DataFrame:
        """
        Shows the Image using matplotlib and returns a list of
        coordinates picked by user.

        Parameters
        ----------
        scale: bool, optional
            Scales the Image if True.

        Returns
        -------
        pd.DataFrame
            List of coordinates selected.
        """
        if scale:
            zscale = ZScaleInterval()
        else:
            def zscale(x):
                return x

        fig, ax = plt.subplots(constrained_layout=True)
        ax.imshow(zscale(self.data()), cmap="Greys_r")
        klkr = clicker(ax, ["source"], markers=["o"])
        plt.show()
        if len(klkr.get_positions()["source"]) == 0:
            return pd.DataFrame([], columns=["xcentroid", "ycentroid"])

        return pd.DataFrame(
            klkr.get_positions()["source"], columns=[
                "xcentroid", "ycentroid"])

    def solve_filed(self) -> Self:
        """
        Creates the WCS headers

        Returns
        -------
        Self
            New `Fits` object that has WCS headers.
        """
        return self

    def zero_correction(self, master_zero: Fits, output: Optional[str] = None,
                        override: bool = False, force: bool = False) -> Self:
        """
        Does zero correction of the data

        Parameters
        ----------
        master_zero : Fits
            Zero file to be used for correction
        output: str, optional
            Path of the new fits file.
        override: bool, default=False
            If True will overwrite the output if a file is already exists.
        force: bool, default=False
            Overcorrection flag

        Returns
        -------
        Self
            Zero corrected `Fits` object

        Raises
        ------
        OverCorrection
            when the `Fits` object is already
            zero corrected and `force` is `False`
        """
        if "SPY_ZERO" not in self.header() or force:
            zero_corrected = subtract_bias(self.ccd(), master_zero.ccd())
            header = self.pure_header()
            header["SPY_ZERO"] = master_zero.file.name

            return self.__class__.from_data_header(
                zero_corrected.data, header=header,
                output=output, override=override
            )

        raise OverCorrection("This Data is already zero corrected")

    def dark_correction(self, master_dark: Fits,
                        exposure: Optional[str] = None,
                        output: Optional[str] = None, override: bool = False,
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
            Path of the new fits file.
        override: bool, default=False
            If True will overwrite the output if a file is already exists.
        force: bool, default=False
            Overcorrection flag

        Returns
        -------
        Self
            Dark corrected `Fits` object

        Raises
        ------
        OverCorrection
            when the `Fits` object is already
            dark corrected and `force` is `False`
        """
        if "SPY_DARK" not in self.header() or force:
            if exposure is None:
                options = {"dark_exposure": 1 * units.s,
                           "data_exposure": 1 * units.s}
            else:
                if exposure not in master_dark.header() or \
                        exposure not in master_dark.header():
                    raise ValueNotFound(
                        f"Key {exposure} not found in file, "
                        "master_dark or both"
                    )

                options = {
                    "dark_exposure": float(
                        master_dark.header()[exposure].values[0]) * units.s,
                    "data_exposure": float(
                        self.header()[exposure].values[0]) * units.s
                }

            dark_corrected = subtract_dark(
                self.ccd(), master_dark.ccd(),
                **options, scale=True
            )
            header = self.pure_header()
            header["SPY_DARK"] = master_dark.file.name

            return self.__class__.from_data_header(
                dark_corrected.data, header=header,
                output=output, override=override
            )

        raise OverCorrection("This Data is already dark corrected")

    def flat_correction(self, master_flat: Fits, output: Optional[str] = None,
                        override: bool = False, force: bool = False) -> Self:
        """
        Does flat correction of the data

        Parameters
        ----------
        master_flat : Fits
            Flat file to be used for correction
        output: str, optional
            Path of the new fits file.
        override: bool, default=False
            If True will overwrite the output if a file is already exists.
        force: bool, default=False
            Overcorrection flag

        Returns
        -------
        Self
            Flat corrected `Fits` object

        Raises
        ------
        OverCorrection
            when the `Fits` object is already
            flat corrected and `force` is `False`
        """
        if "SPY_FLAT" not in self.header() or force:
            flat_corrected = flat_correct(self.ccd(), master_flat.ccd())
            header = self.pure_header()
            header["SPY_FLAT"] = master_flat.file.name

            return self.__class__.from_data_header(
                flat_corrected.data, header=header,
                output=output, override=override
            )

        raise OverCorrection("This Data is already flat corrected")

    def background(self) -> Background:
        """
        Returns a `Background` object of the fits file.
        """
        return Background(self.data())

    def daofind(self, sigma: float = 3, fwhm: float = 3,
                threshold: float = 5) -> pd.DataFrame:
        """
        Runs daofind to detect sources on the image.

        [1]: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html

        [2]: https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html

        Parameters
        ----------
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
        mean, median, std = sigma_clipped_stats(self.data(), sigma=sigma)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
        sources = daofind(self.data() - median)

        if sources is not None:
            return sources.to_pandas()

        return pd.DataFrame(
            [],
            columns=[
                "id", "xcentroid", "ycentroid", "sharpness", "roundness1",
                "roundness2", "npix", "sky", "peak", "flux", "mag"
            ]
        )

    def extract(self, detection_sigma: float = 5,
                min_area: float = 5) -> pd.DataFrame:
        """
        Runs astroalign._find_sources to detect sources on the image.

        Parameters
        ----------
        detection_sigma: float, default=5
            `thresh = detection_sigma * bkg.globalrms`
        min_area: float, default=5
            Minimum area

        Returns
        -------
        pd.DataFrame
            List of sources found on the image.
        """
        bkg = self.background()
        thresh = detection_sigma * bkg.globalrms
        sources = sep_extract(self.data() - bkg.back(), thresh,
                              minarea=min_area)
        sources.sort(order="flux")
        if len(sources) < 0:
            raise NumberOfElementError("No source was found")

        return pd.DataFrame(
            sources,
        ).rename(columns={"x": "xcentroid", "y": "ycentroid"})

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

        table = []

        the_header = self.header()

        if exposure is None:
            exposure_to_use = 0.0
        else:
            if isinstance(exposure, (int, float)):
                exposure_to_use = exposure
            else:
                exposure_to_use = float(the_header[exposure].iloc[0])

        new_xs, new_ys = Fixer.coordinate(xs, ys)
        new_rs = Fixer.aperture(rs)
        new_headers = Fixer.header(headers)

        headers_ = []
        keys_ = []
        for new_header in new_headers:
            keys_.append(new_header)
            try:
                headers_.append(the_header[new_header].iloc[0])
            except KeyError:
                headers_.append(None)

        data = self.data()
        background = self.background()

        clean_d = data - background.rms()
        error = calc_total_error(
            self.data(), background, exposure_to_use
        )
        for new_r in new_rs:
            fluxes, flux_errs, flags = sum_circle(
                data,
                new_xs, new_ys, new_r,
                err=error
            )
            for x, y, flux, flux_err, flag in zip(new_xs, new_ys, fluxes,
                                                  flux_errs, flags):
                value = clean_d[int(x)][int(y)]
                snr = np.nan if value < 0 else math.sqrt(value)
                mag, mag_err = self.flux_to_mag(flux, flux_err,
                                                exposure_to_use)
                table.append(
                    [
                        self.file.name, "sep", x, y, new_r, flux, flux_err,
                        flag, snr, mag, mag_err, *headers_
                    ]
                )
        return pd.DataFrame(
            table,
            columns=[
                "image", "package", "xcentroid", "ycentroid", "aperture",
                "flux", "flux_error", "flag", "snr", "mag", "merr", *keys_
            ]
        ).set_index("image")

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
        table = []

        the_header = self.header()

        if exposure is None:
            exposure_to_use = 0.0
        else:
            if isinstance(exposure, (int, float)):
                exposure_to_use = exposure
            else:
                exposure_to_use = float(the_header[exposure].iloc[0])

        new_xs, new_ys = Fixer.coordinate(xs, ys)
        new_rs = Fixer.aperture(rs)
        new_headers = Fixer.header(headers)

        headers_ = []
        keys_ = []
        for new_header in new_headers:
            keys_.append(new_header)
            try:
                headers_.append(the_header[new_header].iloc[0])
            except KeyError:
                headers_.append(None)

        data = self.data()
        background = self.background()

        clean_d = data - background.rms()

        for new_r in new_rs:
            apertures = CircularAperture([
                [new_x, new_y] for new_x, new_y in zip(new_xs, new_ys)
            ], r=new_r)
            error = calc_total_error(
                self.data(), self.background(), exposure_to_use
            )
            phot_table = aperture_photometry(data, apertures, error=error)

            for phot_line in phot_table:
                value = clean_d[
                    int(phot_line["xcenter"].value)
                ][
                    int(phot_line["ycenter"].value)
                ]
                snr = np.nan if value < 0 else math.sqrt(value)
                mag, mag_err = self.flux_to_mag(
                    phot_line["aperture_sum"],
                    phot_line["aperture_sum_err"],
                    exposure_to_use
                )
                table.append(
                    [
                        self.file.name, "phu",
                        phot_line["xcenter"].value,
                        phot_line["ycenter"].value, new_r,
                        phot_line["aperture_sum"],
                        phot_line["aperture_sum_err"],
                        None, snr, mag, mag_err, *headers_
                    ]
                )

        return pd.DataFrame(
            table,
            columns=[
                "image", "package", "xcentroid", "ycentroid", "aperture",
                "flux", "flux_error", "flag", "snr", "mag", "merr", *keys_
            ]
        ).set_index("image")

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
        return pd.concat(
            (self.photometry_sep(
                xs, ys, rs, headers=headers, exposure=exposure
            ),
             self.photometry_phu(
                 xs, ys, rs, headers=headers, exposure=exposure
             ))
        )

    def shift(self, x: int, y: int, output: Optional[str] = None,
              override: bool = False) -> Self:
        """
        Shifts the data of `Fits` object

        Parameters
        ----------
        x: int
            x coordinate
        y: int
            y coordinate
        output: str, optional
            Path of the new fits file.
        override: bool, default=False
            If True will overwrite the output if a file is already exists.

        Returns
        -------
        Self
            shifted `Fits` object
        """
        shifted_data = np.roll(self.data(), x, axis=1)
        if x < 0:
            shifted_data[:, x:] = 0
        elif x > 0:
            shifted_data[:, 0:x] = 0

        shifted_data = np.roll(shifted_data, y, axis=0)
        if y < 0:
            shifted_data[y:, :] = 0
        elif y > 0:
            shifted_data[0:y, :] = 0

        return self.from_data_header(shifted_data, self.pure_header(),
                                     output=output, override=override)
