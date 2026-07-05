"""
nodal_solver.py
===============
Nodal Analysis solver — finds the operating point where the IPR (inflow)
curve intersects the VLP (outflow) curve.

The nodal point is at the perforations (mid-perforation).
  - IPR gives: reservoir → node (inflow)
  - VLP gives: node → wellhead (outflow)
  - Operating point: q where FBHP_IPR(q) = FBHP_VLP(q)

Also orchestrates the complete analysis and generates all output data
needed by the Streamlit app.

Field units throughout.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple
from .ipr import IPRCalculator
from .vlp import VLPCalculator


class NodalAnalysis:
    """
    Nodal analysis solver combining IPR and VLP.

    Parameters
    ----------
    ipr_calc : IPRCalculator
    vlp_calc : VLPCalculator
    """

    def __init__(self, ipr_calc: IPRCalculator, vlp_calc: VLPCalculator):
        self.ipr = ipr_calc
        self.vlp = vlp_calc

    def operating_point(
        self,
        ipr_model: str = "vogel",
        vlp_correlation: str = "beggs_brill",
        n_points: int = 80,
        tol: float = 1.0,   # psi tolerance
    ) -> dict:
        """
        Find the operating point (intersection of IPR and VLP curves).

        Algorithm:
          1. Compute IPR at n_points from 0 → AOF
          2. Compute VLP at the same flow rates
          3. Find sign change in (IPR_Pwf - VLP_Pwf)
          4. Bisect within that bracket for precision

        Returns
        -------
        dict with:
          q     : operating flow rate (STB/day)
          Pwf   : operating FBHP (psia)
          aof   : Absolute Open Flow (STB/day) at Pwf=0
          found : bool — True if intersection was found
          message : descriptive string
        """
        aof = self.ipr.aof(ipr_model)
        if aof <= 0:
            return {
                "q": 0.0, "Pwf": self.ipr.Pr,
                "aof": 0.0, "found": False,
                "message": "No production: AOF = 0."
            }

        # Build arrays
        q_arr = np.linspace(0.0, aof * 1.05, n_points)
        ipr_pwf = np.array([self.ipr.compute_rate(0.0, ipr_model)] * n_points)  # placeholder
        # IPR: q → Pwf (invert the IPR curve)
        # Build q→Pwf for IPR by interpolation
        q_ipr, pwf_ipr = self.ipr.ipr_curve(ipr_model, n_points * 2)
        # Ensure monotone (q_ipr should be ascending)
        sort_idx = np.argsort(q_ipr)
        q_ipr_s = q_ipr[sort_idx]
        pwf_ipr_s = pwf_ipr[sort_idx]

        # Remove duplicates
        _, uniq = np.unique(q_ipr_s, return_index=True)
        q_ipr_s = q_ipr_s[uniq]
        pwf_ipr_s = pwf_ipr_s[uniq]

        if len(q_ipr_s) < 2:
            return {
                "q": 0.0, "Pwf": self.ipr.Pr,
                "aof": aof, "found": False,
                "message": "IPR curve degenerate."
            }

        # Interpolator for IPR Pwf given q
        ipr_interp = interp1d(
            q_ipr_s, pwf_ipr_s,
            kind="linear",
            bounds_error=False,
            fill_value=(pwf_ipr_s[0], pwf_ipr_s[-1])
        )

        # VLP: compute FBHP at same q points
        q_eval = np.linspace(0.5, aof, n_points)
        vlp_fbhp = np.array([
            self.vlp.compute_fbhp(q, vlp_correlation)["fbhp"]
            for q in q_eval
        ])
        ipr_fbhp = ipr_interp(q_eval)

        # Difference: IPR_Pwf - VLP_Pwf
        diff = ipr_fbhp - vlp_fbhp

        # Find zero crossing
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) == 0:
            # No intersection found — check which side
            if diff[-1] > 0:
                # VLP always below IPR → well flows at AOF (limited by wellbore)
                q_op = aof
                pwf_op = float(ipr_interp(aof))
                msg = "VLP below IPR for all rates: system may be flow-limited."
            else:
                # VLP always above IPR → well cannot produce
                q_op = 0.0
                pwf_op = float(self.ipr.Pr)
                msg = "VLP above IPR for all rates: well cannot produce at these conditions."
            return {
                "q": q_op, "Pwf": pwf_op,
                "aof": aof, "found": False,
                "message": msg,
            }

        # Use the first (lowest-rate) intersection
        idx = sign_changes[0]
        q_lo, q_hi = q_eval[idx], q_eval[idx + 1]

        # Bisection refinement
        for _ in range(60):
            q_mid = (q_lo + q_hi) / 2.0
            diff_mid = float(ipr_interp(q_mid)) - self.vlp.compute_fbhp(q_mid, vlp_correlation)["fbhp"]
            diff_lo  = float(ipr_interp(q_lo))  - self.vlp.compute_fbhp(q_lo,  vlp_correlation)["fbhp"]
            if abs(q_hi - q_lo) < 0.1:
                break
            if diff_lo * diff_mid < 0:
                q_hi = q_mid
            else:
                q_lo = q_mid

        q_op   = (q_lo + q_hi) / 2.0
        pwf_op = self.vlp.compute_fbhp(q_op, vlp_correlation)["fbhp"]

        return {
            "q":    round(q_op, 1),
            "Pwf":  round(pwf_op, 1),
            "aof":  round(aof, 1),
            "found": True,
            "message": f"Operating point: q = {q_op:.0f} STB/day, Pwf = {pwf_op:.0f} psia",
        }

    def full_analysis(
        self,
        ipr_model: str = "vogel",
        vlp_correlation: str = "beggs_brill",
        n_curve_points: int = 80,
    ) -> dict:
        """
        Run complete nodal analysis and return all data for plotting.

        Returns
        -------
        dict with:
          ipr_curves       : dict {model: (q, pwf)} for all models
          vlp_q, vlp_fbhp  : VLP curve arrays
          operating_point  : {q, Pwf, aof, found, message}
          pressure_losses  : breakdown dict at operating point
          reservoir_info   : summary dict
        """
        aof_primary = self.ipr.aof(ipr_model)
        q_max = max(aof_primary * 1.1, 100.0)

        # ---- IPR curves (all models) ---------------------------------
        ipr_curves = self.ipr.all_models_comparison(n_curve_points)

        # ---- VLP curve ----------------------------------------------
        vlp_q, vlp_fbhp = self.vlp.vlp_curve(q_max, vlp_correlation, n_curve_points)

        # ---- Operating point ----------------------------------------
        op = self.operating_point(ipr_model, vlp_correlation)

        # ---- Pressure loss breakdown at operating point --------------
        if op["q"] > 0:
            pl = self.vlp.pressure_loss_breakdown(op["q"], vlp_correlation)
        else:
            pl = {
                "Hydrostatic (Gravity)": 0.0,
                "Friction": 0.0,
                "Acceleration": 0.0,
                "Total ΔP": 0.0,
                "FBHP": self.ipr.Pr,
            }

        # ---- Summary info -------------------------------------------
        reservoir_info = {
            "Reservoir Pressure (psia)":    self.ipr.Pr,
            "Bubble-Point Pressure (psia)": self.ipr.Pb,
            "Productivity Index (STB/d/psi)": self.ipr.PI,
            "AOF (STB/day)":               aof_primary,
            "Operating Rate (STB/day)":     op["q"],
            "Operating FBHP (psia)":        op["Pwf"],
            "Drawdown (psia)":              round(self.ipr.Pr - op["Pwf"], 1),
            "Wellhead Pressure (psia)":     self.vlp.whp,
            "TVD (ft)":                     self.vlp.tvd,
            "Tubing ID (in)":               self.vlp.tubing_id,
        }

        return {
            "ipr_curves":      ipr_curves,
            "vlp_q":           vlp_q,
            "vlp_fbhp":        vlp_fbhp,
            "operating_point": op,
            "pressure_losses": pl,
            "reservoir_info":  reservoir_info,
        }

    def sensitivity_analysis(
        self,
        parameter: str,
        values: list,
        ipr_model: str = "vogel",
        vlp_correlation: str = "beggs_brill",
    ) -> list:
        """
        Run nodal analysis for a range of parameter values.
        Useful for sensitivity plots (e.g., WHP sensitivity).

        Parameters
        ----------
        parameter : 'whp', 'water_cut', 'gor', 'tubing_id'
        values    : list of values to test
        ...

        Returns
        -------
        list of {param_value, q, Pwf}
        """
        results = []
        original_whp = self.vlp.whp
        original_wc  = self.vlp.fp.wc
        original_gor = self.vlp.fp.gor
        original_tid = self.vlp.tubing_id

        for v in values:
            try:
                if parameter == "whp":
                    self.vlp.whp = v
                elif parameter == "water_cut":
                    self.vlp.fp.wc = v
                    # Recompute bubble-point
                    self.vlp.fp.bubble_point = self.vlp.fp._bubble_point_standing(
                        self.vlp.fp.T_res, self.vlp.fp.gor
                    )
                elif parameter == "gor":
                    self.vlp.fp.gor = v
                    self.vlp.fp.bubble_point = self.vlp.fp._bubble_point_standing(
                        self.vlp.fp.T_res, v
                    )
                elif parameter == "tubing_id":
                    self.vlp.tubing_id = v
                    self.vlp.D_ft = v / 12.0
                    import math
                    self.vlp.plc.D_ft = v / 12.0
                    self.vlp.plc.D_in = v
                    self.vlp.plc.A = math.pi * (v / 12.0) ** 2 / 4.0

                op = self.operating_point(ipr_model, vlp_correlation)
                results.append({
                    "param_value": v,
                    "q": op["q"],
                    "Pwf": op["Pwf"],
                    "aof": op["aof"],
                })
            except Exception as e:
                results.append({"param_value": v, "q": 0.0, "Pwf": 0.0, "aof": 0.0})

        # Restore original values
        self.vlp.whp = original_whp
        self.vlp.fp.wc = original_wc
        self.vlp.fp.gor = original_gor
        self.vlp.tubing_id = original_tid
        if hasattr(self.vlp, 'plc'):
            import math
            self.vlp.plc.D_ft = original_tid / 12.0
            self.vlp.plc.D_in = original_tid
            self.vlp.plc.A = math.pi * (original_tid / 12.0) ** 2 / 4.0

        return results
