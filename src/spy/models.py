from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Optional, List, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .fits import Fits

from typing_extensions import Self

import pandas as pd
from astropy.nddata import CCDData

NUMERICS = Union[float, int, List[Union[float, int]]]


class Data(ABC):

    @classmethod
    @abstractmethod
    def sample(cls) -> Self:
        ...

    @abstractmethod
    def header(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def data(self) -> Any:
        ...

    @abstractmethod
    def ccd(self) -> CCDData:
        ...

    @abstractmethod
    def imstat(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def cosmic_clean(self) -> Self:
        ...

    @abstractmethod
    def hedit(self, keys: Union[str, List[str]],
              values: Optional[Union[str, List[str]]] = None,
              delete: Optional[bool] = False,
              value_is_key: bool = False) -> Self:
        ...

    @abstractmethod
    def save_as(self, output: str, override: bool = False) -> Self:
        ...

    @abstractmethod
    def add(self, other: Union[Self, float, int], output: Optional[str] = None,
            override: bool = False) -> Self:
        ...

    @abstractmethod
    def sub(self, other: Union[Self, int, float], output: Optional[str] = None,
            override: bool = False) -> Self:
        ...

    @abstractmethod
    def mul(self, other: Union[Self, int, float], output: Optional[str] = None,
            override: bool = False) -> Self:
        ...

    @abstractmethod
    def div(self, other: Union[Self, int, float], output: Optional[str] = None,
            override: bool = False) -> Self:
        ...

    @abstractmethod
    def imarith(self, other: Union[Self, int, float],
                operand: str,
                output: Optional[str] = None,
                override: bool = False) -> Self:
        ...

    @abstractmethod
    def align(self, reference: Self, output: Optional[str] = None,
              max_control_points: int = 50, min_area: int = 5,
              override: bool = False) -> Self:
        ...

    @abstractmethod
    def show(self, scale: bool = True,
             sources: Optional[pd.DataFrame] = None) -> None:
        ...

    @abstractmethod
    def solve_filed(self) -> Self:
        ...

    @abstractmethod
    def zero_correction(self, master_zero: Self,
                        output: Optional[str] = None,
                        override: bool = True, force: bool = False) -> Self:
        ...

    @abstractmethod
    def dark_correction(self, master_dark: Self,
                        exposure: Optional[str] = None,
                        output: Optional[str] = None, override: bool = False,
                        force: bool = False) -> Self:
        ...

    @abstractmethod
    def flat_correction(self, master_flat: Self, output: Optional[str] = None,
                        override: bool = False, force: bool = False) -> Self:
        ...

    @abstractmethod
    def photometry_sep(self,
                       xs: NUMERICS, ys: NUMERICS, rs: NUMERICS,
                       headers: Optional[Union[str, list[str]]] = None,
                       exposure: Optional[Union[str, float, int]] = None
                       ) -> pd.DataFrame:
        ...


class DataArray(ABC):

    @classmethod
    @abstractmethod
    def sample(cls, numer_of_samples: int = 10) -> Self:
        ...

    @abstractmethod
    def header(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def data(self) -> List[Any]:
        ...

    @abstractmethod
    def ccd(self) -> List[CCDData]:
        ...

    @abstractmethod
    def imstat(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def cosmic_clean(self) -> Self:
        ...

    @abstractmethod
    def hedit(self, keys: Union[str, List[str]],
              values: Optional[Union[str, List[str]]] = None,
              delete: Optional[bool] = False,
              value_is_key: bool = False) -> Self:
        ...

    @abstractmethod
    def hselect(self, fields: Union[str, List[str]]) -> pd.DataFrame:
        ...

    @abstractmethod
    def save_as(self, output: str) -> Self:
        ...

    @abstractmethod
    def add(self, other: Union[
        Self, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        ...

    @abstractmethod
    def sub(self, other: Union[
        Self, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        ...

    @abstractmethod
    def mul(self, other: Union[
        Self, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        ...

    @abstractmethod
    def div(self, other: Union[
        Self, Fits, float, int, List[Union[Fits, float, int]]],
            output: Optional[str] = None) -> Self:
        ...

    @abstractmethod
    def imarith(self, other: Union[
        Self, Fits, float, int, List[Union[Fits, float, int]]], operand: str,
                output: Optional[str] = None) -> Self:
        ...

    @abstractmethod
    def align(self, other: Optional[Union[Fits, int]] = 0,
              output: Optional[str] = None,
              max_control_points: int = 50, min_area: int = 5) -> Self:
        ...

    @abstractmethod
    def show(self, scale: bool = True, interval: float = 1) -> None:
        ...

    @abstractmethod
    def solve_filed(self) -> Self:
        ...

    @abstractmethod
    def zero_correction(self, master_zero: Fits, output: Optional[str] = None,
                        force: bool = False) -> Self:
        ...

    @abstractmethod
    def dark_correction(self, master_dark: Fits,
                        exposure: Optional[str] = None,
                        output: Optional[str] = None,
                        force: bool = False) -> Self:
        ...

    @abstractmethod
    def flat_correction(self, master_flat: Fits, output: Optional[str] = None,
                        force: bool = False) -> Self:
        ...

    @abstractmethod
    def photometry_sep(self,
                       xs: NUMERICS, ys: NUMERICS, rs: NUMERICS,
                       headers: Optional[Union[str, list[str]]] = None,
                       exposure: Optional[Union[str, float, int]] = None
                       ) -> pd.DataFrame:
        ...
