"""Minimal SI unit system for the safe calculator (simple conversions + dimensional checks).

Dimension vector: (m, kg, s, A, K, mol, cd, X) -- X is a symbolic slot used only when
solving a formula for an unknown variable. Anything not in the table is *unsupported*
and the calculator reports ``unknown`` instead of guessing.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

NDIM = 8
DIM_NAMES = ("m", "kg", "s", "A", "K", "mol", "cd", "X")
ZERO = (0,) * NDIM


def _d(m=0, kg=0, s=0, A=0, K=0, mol=0, cd=0) -> tuple:
    return (m, kg, s, A, K, mol, cd, 0)


# symbol -> (factor to SI, dims)
UNITS: dict[str, tuple[float, tuple]] = {
    # length
    "m": (1.0, _d(m=1)), "km": (1e3, _d(m=1)), "dm": (0.1, _d(m=1)), "cm": (1e-2, _d(m=1)), "mm": (1e-3, _d(m=1)),
    "um": (1e-6, _d(m=1)), "µm": (1e-6, _d(m=1)), "μm": (1e-6, _d(m=1)), "nm": (1e-9, _d(m=1)),
    # mass
    "kg": (1.0, _d(kg=1)), "g": (1e-3, _d(kg=1)), "mg": (1e-6, _d(kg=1)), "t": (1e3, _d(kg=1)), "tấn": (1e3, _d(kg=1)),
    "tạ": (100.0, _d(kg=1)), "yến": (10.0, _d(kg=1)),
    # time
    "s": (1.0, _d(s=1)), "ms": (1e-3, _d(s=1)), "min": (60.0, _d(s=1)), "phút": (60.0, _d(s=1)),
    "h": (3600.0, _d(s=1)), "giờ": (3600.0, _d(s=1)), "ngày": (86400.0, _d(s=1)), "năm": (3.1536e7, _d(s=1)),
    # electric current, temperature, amount
    "A": (1.0, _d(A=1)), "mA": (1e-3, _d(A=1)),
    "K": (1.0, _d(K=1)),
    "mol": (1.0, _d(mol=1)), "mmol": (1e-3, _d(mol=1)),
    # derived
    "N": (1.0, _d(m=1, kg=1, s=-2)), "kN": (1e3, _d(m=1, kg=1, s=-2)),
    "J": (1.0, _d(m=2, kg=1, s=-2)), "kJ": (1e3, _d(m=2, kg=1, s=-2)), "MJ": (1e6, _d(m=2, kg=1, s=-2)),
    "cal": (4.184, _d(m=2, kg=1, s=-2)), "kcal": (4184.0, _d(m=2, kg=1, s=-2)),
    "eV": (1.602176634e-19, _d(m=2, kg=1, s=-2)),
    "Wh": (3600.0, _d(m=2, kg=1, s=-2)), "kWh": (3.6e6, _d(m=2, kg=1, s=-2)),
    "W": (1.0, _d(m=2, kg=1, s=-3)), "kW": (1e3, _d(m=2, kg=1, s=-3)), "MW": (1e6, _d(m=2, kg=1, s=-3)),
    "Pa": (1.0, _d(m=-1, kg=1, s=-2)), "kPa": (1e3, _d(m=-1, kg=1, s=-2)), "MPa": (1e6, _d(m=-1, kg=1, s=-2)),
    "atm": (101325.0, _d(m=-1, kg=1, s=-2)), "bar": (1e5, _d(m=-1, kg=1, s=-2)), "mmHg": (133.322, _d(m=-1, kg=1, s=-2)),
    "C": (1.0, _d(s=1, A=1)), "V": (1.0, _d(m=2, kg=1, s=-3, A=-1)), "Ω": (1.0, _d(m=2, kg=1, s=-3, A=-2)),
    "ohm": (1.0, _d(m=2, kg=1, s=-3, A=-2)), "Hz": (1.0, _d(s=-1)),
    "L": (1e-3, _d(m=3)), "l": (1e-3, _d(m=3)), "lít": (1e-3, _d(m=3)), "mL": (1e-6, _d(m=3)), "ml": (1e-6, _d(m=3)),
    "M": (1e3, _d(m=-3, mol=1)),                     # mol/L
    "ha": (1e4, _d(m=2)),
    "%": (0.01, ZERO), "rad": (1.0, ZERO), "vòng": (2 * math.pi, ZERO),
    "°C": (1.0, _d(K=1)), "oC": (1.0, _d(K=1)),      # treated as temperature *interval* inside expressions
}

_SUPER = str.maketrans({"²": "^2", "³": "^3", "⁻": "^-", "¹": "1", "⁴": "^4", "·": "*", "⋅": "*"})
_FACTOR = re.compile(r"^([A-Za-zµμΩ°%àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+?)\^?(-?\d+)?$")


class UnitError(ValueError):
    pass


def dims_add(a: tuple, b: tuple, k: int = 1) -> tuple:
    return tuple(x + k * y for x, y in zip(a, b))


def dims_scale(a: tuple, k: float) -> tuple:
    out = tuple(x * k for x in a)
    if any(abs(v - round(v)) > 1e-9 for v in out):
        raise UnitError("non-integer dimension exponent")
    return tuple(int(round(v)) for v in out)


def parse_unit(u: Optional[str]) -> tuple[float, tuple]:
    """'m/s^2', 'km/h', 'kg*m/s2', 'g/mol', 'mol/L', 'm²' -> (factor_to_SI, dims). Raises UnitError."""
    if u is None:
        return 1.0, ZERO
    s = u.strip().translate(_SUPER).replace(" ", "")
    if s in ("", "1", "-"):
        return 1.0, ZERO
    if s in UNITS:
        return UNITS[s]
    parts = s.split("/")
    factor, dims = 1.0, ZERO
    for i, part in enumerate(parts):
        sign = 1 if i == 0 else -1
        for f in re.split(r"[*.]", part):
            if not f or f == "1":
                continue
            if f in UNITS:
                base, exp = f, 1
            else:
                m = _FACTOR.match(f)
                if not m or m.group(1) not in UNITS:
                    raise UnitError(f"unsupported unit: {f!r} in {u!r}")
                base, exp = m.group(1), int(m.group(2) or 1)
            uf, ud = UNITS[base]
            factor *= uf ** (sign * exp)
            dims = dims_add(dims, ud, sign * exp)
    return factor, dims


def dims_to_str(d: tuple) -> str:
    if all(x == 0 for x in d):
        return "1"
    num = [f"{n}^{e}" if e != 1 else n for n, e in zip(DIM_NAMES, d) if e > 0]
    den = [f"{n}^{-e}" if e != -1 else n for n, e in zip(DIM_NAMES, d) if e < 0]
    s = "*".join(num) or "1"
    return s + ("/" + "/".join(den) if den else "")


_NUM_UNIT = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


@dataclass
class Q:
    value: float           # SI value
    dims: tuple = ZERO


def parse_quantity(x) -> Q:
    """'6 N', '0,33 m/s²', 2, {'value': 2, 'unit': 'kg'} -> Q (SI). Raises UnitError."""
    if isinstance(x, bool):
        raise UnitError("boolean is not a quantity")
    if isinstance(x, (int, float)):
        return Q(float(x), ZERO)
    if isinstance(x, dict):
        v = x.get("value")
        if isinstance(v, str):
            v = float(v.replace(",", "."))
        f, d = parse_unit(x.get("unit"))
        return Q(float(v) * f, d)
    if isinstance(x, str):
        m = _NUM_UNIT.match(x)
        if not m:
            raise UnitError(f"cannot parse quantity {x!r}")
        num = float(m.group(1).replace(",", "."))
        f, d = parse_unit(m.group(2) or None)
        return Q(num * f, d)
    raise UnitError(f"cannot parse quantity {x!r}")


def convert_temperature(value: float, src: str, dst: str) -> Optional[float]:
    s, d = src.replace("oC", "°C"), dst.replace("oC", "°C")
    to_k = {"K": lambda v: v, "°C": lambda v: v + 273.15, "°F": lambda v: (v - 32) * 5 / 9 + 273.15}
    from_k = {"K": lambda v: v, "°C": lambda v: v - 273.15, "°F": lambda v: (v - 273.15) * 9 / 5 + 32}
    if s in to_k and d in from_k:
        return from_k[d](to_k[s](value))
    return None
