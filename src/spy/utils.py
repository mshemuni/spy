from .error import NumberOfElementError
from .models import NUMERICS
from typing import Optional, Union, List, Tuple, Any

import tempfile
from pathlib import Path, PurePath


class Fixer:

    @classmethod
    def fitsify(cls, path: str) -> str:
        """
        adds fits if the given path does not end with either `fit` ot `fits`

        Parameters
        ----------
        path : str
            the path to check

        Returns
        -------
        string
            the same path if it ends with either `fit` of `fits` otherwise
            adds `fits` to the end of the path
        """
        if not (path.endswith("fit") or path.endswith("fits")):
            return f"{path}.fits"

        return path

    @classmethod
    def outputs(cls, output: Optional[str], fits_array) -> Union[List[None], List[str]]:
        """
        Replaces parent directory of the given `fits_array` with the given
        directory `output`. If output is None it will create a temporary one
        in the temp directory

        Parameters
        ----------
        output : str, optional
            directory to replace the parent directory of each file in
            `fits_array`
        fits_array : FitsArray
            `FitsArray` object to change parent directory of each file with
            the given output

        Returns
        -------
        list
            `list` of file paths
        """
        if output is None or not Path(output).is_dir():
            return [None] * len(fits_array)

        to_write = []
        for fits in fits_array:
            f = fits.file
            to_write.append(str(PurePath(output, f.name)))

        return to_write

    @classmethod
    def output(cls, output: Optional[str] = None, override: bool = False,
               prefix: str = "spy_", suffix: str = ".fits",
               fitsify: bool = True) -> str:
        """
        Checks for the `output`. If it's `None` creates a temporary file.

        Parameters
        ----------
        output : str, optional
            output file path
        override : bool
            deletes the existing file if `override` is `True`, otherwise it
            raises an error
        prefix : str
            `prefix` value for created temporary file
        suffix : str
            `suffix` value for created temporary file
        fitsify : bool
            makes sure the file ends with either `fit` or `fits`

        Returns
        -------
        Path
            `path` new file

        Raises
        ------
        FileExistsError
            when file already exists and `override` is `False`
        """
        if output is None:
            with tempfile.NamedTemporaryFile(delete=True, prefix=prefix,
                                             suffix=suffix) as f:
                output = f.name

        if fitsify:
            output = cls.fitsify(output)

        if Path(output).exists():
            if override:
                Path(output).unlink()
            else:
                raise FileExistsError("File already exist")

        return output

    @classmethod
    def aperture(cls, rs: NUMERICS) -> List[Union[float, int]]:
        """
        Makes sure the given aperture(s) are a list of numbers

        Parameters
        ----------
        rs : NUMERICS
            aperture(s)

        Returns
        -------
        list
            Apertures as `list` of numbers. Even if it is just one aperture
        """
        if isinstance(rs, (float, int)):
            rs = [rs]
        return rs

    @classmethod
    def header(cls,
               headers: Optional[Union[str, List[str]]] = None
               ) -> Any:
        """
        Makes sure the given header(s) are a list of headers

        Parameters
        ----------
        headers : Union[str, List[str]], optional
            header(s)

        Returns
        -------
        list
            Headers as `list` of strings. Even if it is just one header
        """

        if headers is None:
            return []

        if isinstance(headers, str):
            headers = [headers]

        return headers

    @classmethod
    def coordinate(cls,
                   xs: NUMERICS, ys: NUMERICS
                   ) -> Tuple[List[Union[float, int]], List[Union[float, int]]]:
        """
        Makes sure the given `x` and `y` coordinate(s) are list of numbers and
        have the same length

        Parameters
        ----------
        xs : NUMERICS
            x coordinate(s)
        ys : NUMERICS
            y coordinate(s)

        Returns
        -------
        Tuple[List[Union[float, int]], List[Union[float, int]]]
            tuple of `xs` and `ys` coordinates

        Raises
        ------
        NumberOfElementError
            when `x` and `y` coordinates does not have the same length
        """
        if isinstance(xs, (float, int)):
            xs = [xs]

        if isinstance(ys, (float, int)):
            ys = [ys]

        if len(xs) != len(ys):
            raise NumberOfElementError("The length of Xs and Ys must be equal")

        return [x for x in xs], [y for y in ys]


class Check:
    @classmethod
    def operand(cls, operand: str) -> None:
        """
        Checks if the operand is both string and one of `["+", "-", "*", "/", "**", "^"]`

        Parameters
        ----------
        operand : str
            the operand
        Returns
        -------
         None


        Raises
        ------
        ValueError
            when operand is not one of `["+", "-", "*", "/", "**", "^"]`
        """
        if operand not in ["+", "-", "*", "/", "**", "^"]:
            raise ValueError("Operand can only be one of these: +, -, *, /")
