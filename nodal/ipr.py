"""
ipr.py
======
Inflow Performance Relationship (IPR) models.

All models compute the relationship between flowing bottomhole pressure
(Pwf) and surface production rate (q) for a given reservoir.

Models implemented:
  1. Linear PI      – single-phase, above bubble-point
  2. Vogel           – solution-gas drive (below bubble-point)
  3. Fetkovitch      – high-velocity / turbulence (Jones-Blount-Glaze)
  4. Standing        – modified Vogel with partial penetration efficiency
  5. Jones (composite) – Darcy + non-Darcy turbulence

Field units throughout:
  Pressure   : psia
  Rate       : STB/day
  PI         : STB/day/psi
"""

import numpy as np
from typing import Tuple


class IPRCalculator:
    """
    Compute IPR curves for a reservoir.

    Parameters
    ----------
    reservoir_pressure : float
        Static reservoir pressure Pr (psia)
    productivity_index : float
        Productivity index J (STB/day/psi)  – used by Linear PI & Standing
    bubble_point : float
        Bubble-point pressure Pb (psia)
    reservoir_temp : float
        Reservoir temperature (°F) – used for Fetkovitch
    jones_a : float, optional
        Jones turbulence coefficient A (psi²/cp / (STB/day)²) – Jones model only
    jones_b : float, optional
        Jones Darcy coefficient B (psi²/cp / (STB/day)) – Jones model only
    fetkovitch_n : float, optional
        Fetkovitch flow exponent n (0.5 – 1.0, default 1.0 = Darcy)
    completion_efficiency : float, optional
        Standing completion efficiency FE (0 – 1.5, default 1.0)
    """

    def __init__(
        self,
        reservoir_pressure: float,
        productivity_index: float,
        bubble_point: float,
        reservoir_temp: float = 180.0,
        jones_a: float = 0.0,
        jones_b: float = None,
        fetkovitch_n: float = 1.0,
        completion_efficiency: float = 1.0,
    ):
        self.Pr = reservoir_pressure
        self.PI = productivity_index
        self.Pb = bubble_point
        self.T = reservoir_temp
        self.jones_a = jones_a
        self.jones_b = jones_b if jones_b is not None else productivity_index
        self.n = max(0.5, min(1.0, fetkovitch_n))
        self.FE = completion_efficiency

        # AOF (Absolute Open Flow) at Pwf = 0 for the selected model
        self._aof_cache = {}

    # ------------------------------------------------------------------ #
    # 1. LINEAR PI MODEL                                                   #
    # ------------------------------------------------------------------ #
    def _linear_pi(self, Pwf: float) -> float:
        """q = PI × (Pr − Pwf). Valid above and below bubble point."""
        if Pwf >= self.Pr:
            return 0.0
        return max(0.0, self.PI * (self.Pr - Pwf))

    # ------------------------------------------------------------------ #
    # 2. VOGEL MODEL (1968)                                                #
    # ------------------------------------------------------------------ #
    def _vogel(self, Pwf: float) -> float:
        """
        Vogel (1968) IPR for solution-gas drive reservoirs.
        qo_max estimated from PI at bubble-point or above.
        """
        if Pwf >= self.Pr:
            return 0.0
        qo_max = self._vogel_aof()
        x = Pwf / self.Pr
        q = qo_max * (1.0 - 0.2 * x - 0.8 * x ** 2)
        return max(0.0, q)

    def _vogel_aof(self) -> float:
        """Max AOF for Vogel at Pwf=0."""
        if "vogel" in self._aof_cache:
            return self._aof_cache["vogel"]
        # Use PI to anchor: at Pwf=Pb, q_Pb = PI*(Pr-Pb) if Pr>Pb, else linear
        if self.Pr > self.Pb > 0:
            q_at_pb = self.PI * (self.Pr - self.Pb)
            # Vogel fraction at Pwf=Pb
            x = self.Pb / self.Pr
            vogel_frac = 1.0 - 0.2 * x - 0.8 * x ** 2
            qo_max = q_at_pb / vogel_frac if vogel_frac > 0 else self.PI * self.Pr
        else:
            # No gas: pure linear
            qo_max = self.PI * self.Pr
        self._aof_cache["vogel"] = qo_max
        return qo_max

    # ------------------------------------------------------------------ #
    # 3. FETKOVITCH MODEL (1973)                                           #
    # ------------------------------------------------------------------ #
    def _fetkovitch(self, Pwf: float) -> float:
        """
        Fetkovitch (1973): q = C × (Pr² − Pwf²)^n
        C is derived from PI at pseudo-steady-state.
        n = flow exponent (0.5 = fully turbulent, 1.0 = Darcy)
        """
        if Pwf >= self.Pr:
            return 0.0
        # Derive C from PI: at Pwf near Pr, dq/dPwf ≈ PI
        # PI ≈ C × n × (Pr²-Pwf²)^(n-1) × 2Pwf → at Pwf→Pr: indeterminate
        # Use: q(Pwf=0) = PI*Pr from linear → C = PI*Pr / Pr^(2n)
        C = self.PI * self.Pr / (self.Pr ** (2 * self.n))
        dp2 = max(0.0, self.Pr ** 2 - Pwf ** 2)
        q = C * dp2 ** self.n
        return max(0.0, q)

    # ------------------------------------------------------------------ #
    # 4. STANDING MODEL (modified Vogel with Efficiency, 1970)             #
    # ------------------------------------------------------------------ #
    def _standing(self, Pwf: float) -> float:
        """
        Standing (1970) modified Vogel with completion efficiency FE.
        Vogel equation with FE factor on the linear + quadratic terms.
        """
        if Pwf >= self.Pr:
            return 0.0
        qo_max = self._standing_aof()
        x = Pwf / self.Pr
        q = qo_max * (1.0 - 0.2 * (x / self.FE) - 0.8 * (x / self.FE) ** 2)
        return max(0.0, q)

    def _standing_aof(self) -> float:
        """AOF for Standing model."""
        if "standing" in self._aof_cache:
            return self._aof_cache["standing"]
        # Same anchoring as Vogel but with FE
        if self.Pr > self.Pb > 0:
            q_at_pb = self.PI * (self.Pr - self.Pb)
            x = self.Pb / self.Pr
            standing_frac = 1.0 - 0.2 * (x / self.FE) - 0.8 * (x / self.FE) ** 2
            qo_max = q_at_pb / max(standing_frac, 0.01)
        else:
            qo_max = self.PI * self.Pr * self.FE
        self._aof_cache["standing"] = qo_max
        return qo_max

    # ------------------------------------------------------------------ #
    # 5. JONES COMPOSITE MODEL (Jones, Blount & Glaze, 1976)               #
    # ------------------------------------------------------------------ #
    def _jones(self, Pwf: float) -> float:
        """
        Jones composite model: Pr − Pwf = A×q² + B×q
        A = non-Darcy (turbulence) coefficient [psi/(STB/day)²]
        B = Darcy coefficient [psi/(STB/day)]
        
        Solved by quadratic formula: q = (−B + sqrt(B²+4A×ΔP)) / (2A)
        """
        if Pwf >= self.Pr:
            return 0.0
        dP = self.Pr - Pwf
        A = self.jones_a
        B = self.jones_b if self.jones_b else 1.0 / self.PI
        if abs(A) < 1e-12:
            # Pure Darcy (B only)
            return max(0.0, dP / B)
        discriminant = B ** 2 + 4.0 * A * dP
        q = (-B + discriminant ** 0.5) / (2.0 * A)
        return max(0.0, q)

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #
    def compute_rate(self, Pwf: float, model: str = "vogel") -> float:
        """
        Compute flow rate for a given Pwf.

        Parameters
        ----------
        Pwf   : bottomhole flowing pressure (psia)
        model : one of 'linear_pi', 'vogel', 'fetkovitch', 'standing', 'jones'

        Returns
        -------
        q : flow rate (STB/day)
        """
        model = model.lower().replace(" ", "_")
        dispatch = {
            "linear_pi": self._linear_pi,
            "linear pi": self._linear_pi,
            "vogel": self._vogel,
            "fetkovitch": self._fetkovitch,
            "standing": self._standing,
            "jones": self._jones,
        }
        fn = dispatch.get(model)
        if fn is None:
            raise ValueError(f"Unknown IPR model: '{model}'. "
                             f"Choose from: {list(dispatch.keys())}")
        return fn(Pwf)

    def ipr_curve(
        self, model: str = "vogel", n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a full IPR curve (q, Pwf arrays).

        Returns
        -------
        q_array    : np.ndarray of flow rates (STB/day), high→low
        pwf_array  : np.ndarray of BHFPs (psia), high→low
        """
        pwf_array = np.linspace(self.Pr, 0.0, n_points)
        q_array = np.array([self.compute_rate(p, model) for p in pwf_array])
        return q_array, pwf_array

    def aof(self, model: str = "vogel") -> float:
        """Absolute Open Flow at Pwf = 0 for the given model."""
        return self.compute_rate(0.0, model)

    def all_models_comparison(
        self, n_points: int = 100
    ) -> dict:
        """
        Return IPR curves for all 5 models.
        Returns dict: {model_name: (q_array, pwf_array)}
        """
        models = ["linear_pi", "vogel", "fetkovitch", "standing", "jones"]
        results = {}
        for m in models:
            results[m] = self.ipr_curve(m, n_points)
        return results
