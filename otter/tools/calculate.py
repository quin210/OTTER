"""`calculate`: constrained AST interpreter with units (no eval / exec / imports / attributes).

Supports + - * / // % **, unary +/-, parentheses, numeric literals, named variables (with
units) and a whitelist of math functions/constants. Units are tracked as SI dimension
vectors; adding incompatible dimensions, unsupported units or impossible conversions give
``status="unknown"`` (never a guessed number). Formula mode substitutes into a formula
fetched from an ontology rule (``rule_id``), solving numerically for the target if needed.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Optional

from otter.tools.units import (NDIM, ZERO, Q, UnitError, convert_temperature, dims_scale, dims_to_str, parse_quantity,
                               parse_unit)

MAX_EXPR_CHARS = 400
MAX_NODES = 200
MAX_POW = 64
MAX_ABS = 1e300

_FUNCS_DIMLESS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "radians": math.radians, "degrees": math.degrees, "factorial": None,
}
_CONSTS = {"pi": math.pi, "e": math.e, "g": None}   # "g" is NOT a constant (grams / gravity ambiguity)


class CalcError(ValueError):
    def __init__(self, code: str, msg: str = ""):
        super().__init__(msg or code)
        self.code = code


def _sin_deg(x):  # helpers for degree-based trig commonly used in Vietnamese physics
    return math.sin(math.radians(x))


class _Interp:
    def __init__(self, env: dict[str, Q]):
        self.env = env
        self.nodes = 0
        self.trace: list[str] = []

    def run(self, expr: str) -> Q:
        if len(expr) > MAX_EXPR_CHARS:
            raise CalcError("expression_too_long")
        try:
            tree = ast.parse(expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-"), mode="eval")
        except SyntaxError as e:
            raise CalcError("syntax_error", str(e))
        return self.ev(tree.body)

    def ev(self, n: ast.AST) -> Q:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise CalcError("expression_too_complex")
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise CalcError("disallowed_literal", repr(n.value))
            return Q(float(n.value), ZERO)
        if isinstance(n, ast.Name):
            if n.id in self.env:
                return self.env[n.id]
            if n.id in _CONSTS and _CONSTS[n.id] is not None:
                return Q(_CONSTS[n.id], ZERO)
            raise CalcError("unknown_variable", n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = self.ev(n.operand)
            return Q(-v.value if isinstance(n.op, ast.USub) else v.value, v.dims)
        if isinstance(n, ast.BinOp):
            a, b = self.ev(n.left), self.ev(n.right)
            op = n.op
            if isinstance(op, (ast.Add, ast.Sub)):
                if a.dims != b.dims:
                    raise CalcError("dimension_mismatch", f"{dims_to_str(a.dims)} vs {dims_to_str(b.dims)}")
                return self._chk(Q(a.value + b.value if isinstance(op, ast.Add) else a.value - b.value, a.dims))
            if isinstance(op, ast.Mult):
                return self._chk(Q(a.value * b.value, tuple(x + y for x, y in zip(a.dims, b.dims))))
            if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if b.value == 0:
                    raise CalcError("division_by_zero")
                if isinstance(op, ast.Div):
                    return self._chk(Q(a.value / b.value, tuple(x - y for x, y in zip(a.dims, b.dims))))
                if a.dims != ZERO or b.dims != ZERO:
                    raise CalcError("dimension_mismatch", "// and % need dimensionless operands")
                return Q(a.value // b.value if isinstance(op, ast.FloorDiv) else a.value % b.value, ZERO)
            if isinstance(op, ast.Pow):
                if b.dims != ZERO:
                    raise CalcError("dimension_mismatch", "exponent must be dimensionless")
                if abs(b.value) > MAX_POW:
                    raise CalcError("exponent_too_large")
                try:
                    dims = dims_scale(a.dims, b.value) if a.dims != ZERO else ZERO
                except UnitError:
                    raise CalcError("dimension_mismatch", "non-integer power of a dimensional quantity")
                try:
                    return self._chk(Q(float(a.value ** b.value), dims))
                except (OverflowError, ZeroDivisionError, TypeError) as e:
                    raise CalcError("math_domain_error", str(e))
            raise CalcError("disallowed_operator", type(op).__name__)
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.keywords:
                raise CalcError("disallowed_call")
            fname = n.func.id
            args = [self.ev(x) for x in n.args]
            return self._call(fname, args)
        raise CalcError("disallowed_syntax", type(n).__name__)

    def _call(self, f: str, args: list[Q]) -> Q:
        try:
            if f in ("sqrt", "cbrt"):
                if len(args) != 1:
                    raise CalcError("bad_arity", f)
                k = 0.5 if f == "sqrt" else 1 / 3
                try:
                    dims = dims_scale(args[0].dims, k)
                except UnitError:
                    raise CalcError("dimension_mismatch", f"{f} of {dims_to_str(args[0].dims)}")
                if f == "sqrt" and args[0].value < 0:
                    raise CalcError("math_domain_error", "sqrt of negative")
                v = math.sqrt(args[0].value) if f == "sqrt" else math.copysign(abs(args[0].value) ** (1 / 3), args[0].value)
                return Q(v, dims)
            if f == "abs" and len(args) == 1:
                return Q(abs(args[0].value), args[0].dims)
            if f in ("min", "max") and args:
                if len({a.dims for a in args}) != 1:
                    raise CalcError("dimension_mismatch", f)
                pick = min if f == "min" else max
                return Q(pick(a.value for a in args), args[0].dims)
            if f == "round" and 1 <= len(args) <= 2:
                nd = int(args[1].value) if len(args) == 2 else 0
                return Q(round(args[0].value, nd), args[0].dims)
            if f in ("sind", "cosd", "tand"):
                if len(args) != 1 or args[0].dims != ZERO:
                    raise CalcError("dimension_mismatch", f)
                fn = {"sind": math.sin, "cosd": math.cos, "tand": math.tan}[f]
                return Q(fn(math.radians(args[0].value)), ZERO)
            if f in _FUNCS_DIMLESS and _FUNCS_DIMLESS[f] is not None:
                if len(args) not in (1, 2) or any(a.dims != ZERO for a in args):
                    raise CalcError("dimension_mismatch", f"{f} needs dimensionless argument")
                return Q(_FUNCS_DIMLESS[f](*[a.value for a in args]), ZERO)
        except (ValueError, OverflowError) as e:
            if isinstance(e, CalcError):
                raise
            raise CalcError("math_domain_error", str(e))
        raise CalcError("disallowed_function", f)

    @staticmethod
    def _chk(q: Q) -> Q:
        if not math.isfinite(q.value) or abs(q.value) > MAX_ABS:
            raise CalcError("non_finite_result")
        return q


def _sig(x: float, digits: int) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (digits - 1))


def _parse_env(variables: Optional[dict[str, Any]]) -> tuple[dict[str, Q], list[str]]:
    env: dict[str, Q] = {}
    trace = []
    for k, v in (variables or {}).items():
        if not str(k).isidentifier():
            raise CalcError("bad_variable_name", str(k))
        env[k] = parse_quantity(v)
        trace.append(f"{k} = {v}  (SI: {env[k].value:g} {dims_to_str(env[k].dims)})")
    return env, trace


def _declared_unit(v: Any) -> Optional[str]:
    """'geography:c:x (°C)' -> '°C'; 'm/s^2' -> 'm/s^2'; unparseable -> None."""
    if not isinstance(v, str):
        return None
    m = re.search(r"\(([^()]+)\)\s*$", v)
    cand = m.group(1).strip() if m else v.strip()
    try:
        parse_unit(cand)
        return cand
    except UnitError:
        return None


def _to_declared(env: dict[str, Q], declared: dict[str, str], trace: list[str]) -> dict[str, Q]:
    out = {}
    for k, q in env.items():
        if k not in declared:
            out[k] = q
            continue
        f, d = parse_unit(declared[k])
        if q.dims == ZERO and d != ZERO:
            trace.append(f"{k}: no unit given, assumed {declared[k]}")
            out[k] = Q(q.value, ZERO)
        elif q.dims != d:
            raise CalcError("dimension_mismatch", f"{k} has {dims_to_str(q.dims)}, formula expects {declared[k]}")
        else:
            out[k] = Q(q.value / f, ZERO)
            trace.append(f"{k} = {out[k].value:g} {declared[k]}")
    return out


def _split_formula(formula: str) -> tuple[str, str]:
    f = formula.replace("==", "=")
    if f.count("=") != 1:
        raise CalcError("formula_not_an_equation", formula)
    lhs, rhs = (s.strip() for s in f.split("="))
    return lhs, rhs


def _solve(lhs: str, rhs: str, target: str, env: dict[str, Q]) -> tuple[float, Optional[tuple]]:
    """Solve lhs(target)=rhs(target) numerically (secant/bisection on SI values).
    Returns (si_value, dims or None if dims cannot be derived)."""
    # dims: give the target the symbolic X dimension and read off the balance
    sym = dict(env)
    sym[target] = Q(1.0, (0,) * (NDIM - 1) + (1,))
    dims = None
    try:
        L, R = _Interp(sym).run(lhs), _Interp(sym).run(rhs)
        kl, kr = L.dims[-1], R.dims[-1]
        if kl != kr:
            base = tuple((r - l) / (kl - kr) for l, r in zip(L.dims[:-1], R.dims[:-1]))
            if all(abs(b - round(b)) < 1e-9 for b in base):
                dims = tuple(int(round(b)) for b in base) + (0,)
    except CalcError:
        dims = None

    def f(x: float) -> float:
        e = dict(env)
        e[target] = Q(x, dims or ZERO)
        try:
            a = _Interp(e).run(lhs)
            b = _Interp(e).run(rhs)
        except CalcError as ce:
            if ce.code == "dimension_mismatch" and dims is None:
                raise CalcError("unknown_dimension", "cannot derive target dimension")
            raise
        return a.value - b.value

    x0, x1 = 1.0, 2.0
    f0, f1 = f(x0), f(x1)
    for _ in range(100):
        if abs(f1) < 1e-12 * max(1.0, abs(x1)):
            return x1, dims
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
    if abs(f1) < 1e-9 * max(1.0, abs(x1)):
        return x1, dims
    raise CalcError("no_solution_found")


def calculate(ctx: Any, expression: Optional[str] = None, variables: Optional[dict[str, Any]] = None,
              target_unit: Optional[str] = None, rule_id: Optional[str] = None, target: Optional[str] = None,
              convert: Optional[dict[str, Any]] = None, precision: int = 6,
              compare_with: Optional[dict[str, str]] = None, tolerance: float = 0.02) -> dict[str, Any]:
    trace: list[str] = []
    out: dict[str, Any] = {"status": "error", "value": None, "unit": None, "trace": trace}
    try:
        precision = max(1, min(int(precision or 6), 15))
        # --- plain unit conversion ------------------------------------------------
        if convert:
            v = float(str(convert.get("value")).replace(",", "."))
            src, dst = convert.get("from_unit"), convert.get("to_unit")
            t = convert_temperature(v, src or "", dst or "")
            if t is not None:
                out.update(status="ok", value=_sig(t, precision), unit=dst, operation="convert")
                trace.append(f"{v} {src} -> {out['value']} {dst} (affine temperature conversion)")
                return out
            fs, ds = parse_unit(src)
            fd, dd = parse_unit(dst)
            if ds != dd:
                out.update(status="unknown", reason=f"incompatible_units({dims_to_str(ds)} vs {dims_to_str(dd)})")
                return out
            r = v * fs / fd
            out.update(status="ok", value=_sig(r, precision), unit=dst, operation="convert")
            trace.append(f"{v} {src} = {v * fs:g} SI = {out['value']} {dst}")
            return out

        env, vtrace = _parse_env(variables)
        trace.extend(vtrace)
        formula_used = None
        # --- formula substitution from an ontology rule ---------------------------
        if rule_id:
            rule = ctx.store.get_rule(rule_id) if ctx is not None else None
            if rule is None:
                out.update(status="error", reason="unknown_rule_id")
                return out
            if not rule.formula:
                out.update(status="unknown", reason="rule_has_no_formula")
                return out
            formula_used = rule.formula
            out["rule_id"] = rule_id
            out["formula_source_span_ids"] = list(rule.source_span_ids)
            lhs, rhs = _split_formula(rule.formula)
            trace.append(f"formula from {rule_id}: {rule.formula}")
            if target is None and lhs.isidentifier() and lhs not in env:
                target = lhs
            if target is None:
                out.update(status="error", reason="target_variable_required")
                return out
            declared = {k: _declared_unit(v) for k, v in (rule.variables or {}).items()}
            declared = {k: v for k, v in declared.items() if v}
            if declared:
                # textbook formulas often carry units in their coefficients: substitute numbers
                # expressed in the declared units, then attach the declared unit of the target.
                env = _to_declared(env, declared, trace)
            if target == lhs and target not in env:
                q = _Interp(env).run(rhs)
                trace.append(f"{target} = {rhs} -> {q.value:g}")
            else:
                si, dims = _solve(lhs, rhs, target, env)
                trace.append(f"solved {rule.formula} for {target}: {si:g}")
                q = Q(si, dims or ZERO)
                if dims is None and not declared:
                    out.update(status="unknown", reason="cannot_determine_result_dimension", value_si=_sig(si, precision))
                    return out
            if declared:
                if q.dims != ZERO:
                    out.update(status="unknown", reason="dimension_mismatch_in_declared_unit_formula")
                    return out
                tu = declared.get(target)
                if tu is None:
                    out.update(status="unknown", reason="target_unit_not_declared", value=_sig(q.value, precision))
                    return out
                f, d = parse_unit(tu)
                q = Q(q.value * f, d)
                trace.append(f"{target} in declared unit {tu}")
                target_unit = target_unit or tu
            return _finish(out, q, target_unit, precision, trace, compare_with, tolerance, formula_used)
        if not expression:
            out.update(status="error", reason="no_expression")
            return out
        it = _Interp(env)
        q = it.run(str(expression))
        trace.append(f"{expression} -> {q.value:g} {dims_to_str(q.dims)} (SI)")
        return _finish(out, q, target_unit, precision, trace, compare_with, tolerance, formula_used)
    except UnitError as e:
        out.update(status="unknown", reason=f"unsupported_unit: {e}")
        return out
    except CalcError as e:
        status = "unknown" if e.code in ("dimension_mismatch", "unknown_dimension") else "error"
        out.update(status=status, reason=f"{e.code}: {e}" if str(e) != e.code else e.code)
        return out
    except (TypeError, ValueError) as e:
        out.update(status="error", reason=f"bad_input: {e}")
        return out


def _finish(out, q: Q, target_unit, precision, trace, compare_with, tolerance, formula_used):
    if target_unit:
        try:
            f, d = parse_unit(target_unit)
        except UnitError as e:
            out.update(status="unknown", reason=f"unsupported_unit: {e}")
            return out
        if d != q.dims:
            out.update(status="unknown", reason=f"dimension_mismatch(result {dims_to_str(q.dims)} vs {target_unit})",
                       value_si=_sig(q.value, precision))
            return out
        val, unit = q.value / f, target_unit
    else:
        val, unit = q.value, (dims_to_str(q.dims) if q.dims != ZERO else None)
    val = _sig(val, precision)
    trace.append(f"result = {val} {unit or ''}".strip())
    out.update(status="ok", value=val, unit=unit, dims=dims_to_str(q.dims), si_value=_sig(q.value, precision))
    if formula_used:
        out["formula"] = formula_used
    if compare_with:
        cmp = {}
        for k, txt in compare_with.items():
            try:
                c = parse_quantity(txt)
            except UnitError:
                cmp[k] = "unknown"
                continue
            if c.dims != q.dims and c.dims != ZERO:
                cmp[k] = "mismatch_dimension"
                continue
            ref = c.value if c.dims == q.dims else c.value * (parse_unit(unit)[0] if target_unit else 1.0)
            denom = max(abs(q.value), 1e-12)
            cmp[k] = "match" if abs(ref - q.value) / denom <= tolerance else "mismatch"
        out["comparison"] = cmp
    return out
