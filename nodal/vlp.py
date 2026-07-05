"""
vlp.py
======
Vertical Lift Performance (VLP) / Tubing Performance Curve (TPC) calculations.

Integrates the multiphase pressure gradient from the wellhead downward
to compute the Flowing Bottom-Hole Pressure (FBHP) for a given surface rate.
This builds the Outflow (VLP) curve for nodal analysis.

Method:
  Numerically integrate dP/dL from wellhead to perforations using
  the selected correlation (Beggs & Brill or Hagedorn & Brown).
  Temperature is assumed to vary linearly from wellhead to reservoir.

Field units:
  Depth      : ft (True Vertical Depth, TVD)
  Pressure   : psia
  Rate       : STB/day (total liquid)
  Temperature: °F
"""

import numpy as np
from .pressure_loss import PressureLossCalculator


# Standard API tubing sizes: {name: ID in inches}
STANDARD_TUBING_SIZES = {
    "2-3/8\" (1.995\" ID)": 1.995,
    "2-7/8\" (2.441\" ID)": 2.441,
    "3-1/2\" (2.992\" ID)": 2.992,
    "4\" (3.476\" ID)": 3.476,
    "4-1/2\" (3.958\" ID)": 3.958,
}


class VLPCalculator:
    """
    Compute the VLP (outflow) curve for a well.

    Parameters
    ----------
    fluid_props : FluidProperties
        PVT fluid properties object
    tvd : float
        True vertical depth to mid-perforation (ft)
    md : float
        Measured depth to mid-perforation (ft)
    tubing_id : float
        Tubing inner diameter (inches)
    wellhead_pressure : float
        Flowing wellhead (tubing head) pressure (psia), default 100
    reservoir_temp : float
        Temperature at perforations (°F)
    surface_temp : float
        Temperature at wellhead (°F), default 75
    roughness : float
        Pipe wall roughness (ft), default 0.0006
    n_segments : int
        Number of integration segments, default 50
    """

    def __init__(
        self,
        fluid_props,
        tvd: float,
        md: float,
        tubing_id: float,
        wellhead_pressure: float = 100.0,
        reservoir_temp: float = 180.0,
        surface_temp: float = 75.0,
        roughness: float = 0.0006,
        n_segments: int = 50,
    ):
        self.fp = fluid_props
        self.tvd = tvd
        self.md = md
        self.tubing_id = tubing_id
        self.whp = wellhead_pressure
        self.T_res = reservoir_temp
        self.T_surf = surface_temp
        self.roughness = roughness
        self.n_seg = n_segments

        # Average inclination (degrees from vertical)
        # For deviated wells, sin(avg inclination) corrects hydrostatic
        self.avg_inclination = self._avg_inclination()

        self.plc = PressureLossCalculator(
            fluid_props=self.fp,
            tubing_id=self.tubing_id,
            roughness=self.roughness,
            inclination=self.avg_inclination,
        )

    def _avg_inclination(self) -> float:
        """Compute average deviation angle (degrees from vertical)."""
        if self.md <= self.tvd or self.md < 1.0:
            return 0.0
        # Simple: average inclination = acos(TVD/MD)
        import math
        ratio = min(1.0, self.tvd / self.md)
        return math.degrees(math.acos(ratio))

    def _temperature_at_depth(self, depth_ft: float) -> float:
        """
        Linear temperature profile from wellhead (surface_temp) to TVD (reservoir_temp).
        depth_ft is measured from surface (0 = surface, tvd = reservoir).
        """
        if self.tvd <= 0:
            return self.T_res
        frac = depth_ft / self.tvd
        return self.T_surf + frac * (self.T_res - self.T_surf)

    def compute_fbhp(
        self,
        q_liq: float,
        correlation: str = "beggs_brill",
    ) -> dict:
        """
        Compute FBHP at perforations for a given surface liquid rate.

        Integrates pressure from wellhead → perforations in n_seg steps.
        Returns dict with:
          fbhp        : flowing BHP at perforations (psia)
          dP_gravity  : total hydrostatic pressure loss (psi)
          dP_friction : total friction pressure loss (psi)
          dP_accel    : total acceleration pressure loss (psi)
          dP_total    : total pressure loss (psi)
          profile     : list of (depth, pressure) tuples for plotting
        """
        # Build depth array from wellhead (0) to TVD
        depths = np.linspace(0.0, self.tvd, self.n_seg + 1)
        dz = self.tvd / self.n_seg   # depth increment per segment (ft)

        P_current = self.whp   # start at wellhead
        dP_grav_total = 0.0
        dP_fric_total = 0.0
        dP_acc_total  = 0.0
        profile = [(0.0, P_current)]

        for i in range(self.n_seg):
            depth_mid = (depths[i] + depths[i + 1]) / 2.0
            T_mid = self._temperature_at_depth(depth_mid)
            P_mid = max(14.7, P_current)

            if correlation == "beggs_brill":
                grad = self.plc.beggs_brill_gradient(q_liq, P_mid, T_mid)
            elif correlation == "hagedorn_brown":
                grad = self.plc.hagedorn_brown_gradient(q_liq, P_mid, T_mid)
            else:
                raise ValueError(f"Unknown correlation: '{correlation}'")

            dP_g = grad["gravity_gradient"]  * dz
            dP_f = grad["friction_gradient"] * dz
            dP_a = grad["accel_gradient"]    * dz

            P_current += dP_g + dP_f + dP_a
            P_current = max(P_current, 14.7)

            dP_grav_total += dP_g
            dP_fric_total += dP_f
            dP_acc_total  += dP_a

            profile.append((depths[i + 1], P_current))

        fbhp = P_current
        dP_total = fbhp - self.whp

        return {
            "fbhp": fbhp,
            "dP_gravity":  dP_grav_total,
            "dP_friction": dP_fric_total,
            "dP_accel":    dP_acc_total,
            "dP_total":    dP_total,
            "profile": profile,
        }

    def vlp_curve(
        self,
        q_max: float,
        correlation: str = "beggs_brill",
        n_points: int = 60,
    ) -> tuple:
        """
        Generate the full VLP curve (q → FBHP).

        Parameters
        ----------
        q_max        : maximum flow rate to evaluate (STB/day)
        correlation  : 'beggs_brill' or 'hagedorn_brown'
        n_points     : number of points

        Returns
        -------
        (q_array, fbhp_array) as numpy arrays
        """
        # Use a log-scale so we capture the curve knee well
        q_min = max(1.0, q_max * 0.005)
        q_vals = np.concatenate([
            np.linspace(0.0, q_min, 3),
            np.geomspace(q_min, q_max, n_points - 3)
        ])
        q_vals = np.unique(q_vals)

        fbhp_vals = []
        for q in q_vals:
            if q <= 0:
                # At q=0: FBHP = hydrostatic head only
                res = self.compute_fbhp(1e-3, correlation)
                # For zero flow, friction = 0, only gravity
                fbhp_vals.append(self.whp + res["dP_gravity"])
            else:
                res = self.compute_fbhp(q, correlation)
                fbhp_vals.append(res["fbhp"])

        return np.array(q_vals), np.array(fbhp_vals)

    def multi_tubing_vlp(
        self,
        tubing_sizes: dict,
        q_max: float,
        correlation: str = "beggs_brill",
        n_points: int = 60,
    ) -> dict:
        """
        Compute VLP curves for multiple tubing sizes.

        Parameters
        ----------
        tubing_sizes : dict mapping name → inner diameter (inches)
        q_max        : maximum surface rate (STB/day)
        correlation  : 'beggs_brill' or 'hagedorn_brown'

        Returns
        -------
        dict: {tubing_name: (q_array, fbhp_array)}
        """
        results = {}
        for name, id_in in tubing_sizes.items():
            calc = VLPCalculator(
                fluid_props=self.fp,
                tvd=self.tvd,
                md=self.md,
                tubing_id=id_in,
                wellhead_pressure=self.whp,
                reservoir_temp=self.T_res,
                surface_temp=self.T_surf,
                roughness=self.roughness,
                n_segments=self.n_seg,
            )
            q_arr, fbhp_arr = calc.vlp_curve(q_max, correlation, n_points)
            results[name] = (q_arr, fbhp_arr)
        return results

    def pressure_loss_breakdown(
        self,
        q_liq: float,
        correlation: str = "beggs_brill",
    ) -> dict:
        """
        Detailed pressure loss breakdown at a specific flow rate.
        Returns gravity, friction, acceleration, and total losses (psi).
        """
        res = self.compute_fbhp(q_liq, correlation)
        return {
            "Hydrostatic (Gravity)": res["dP_gravity"],
            "Friction":              res["dP_friction"],
            "Acceleration":          res["dP_accel"],
            "Total ΔP":              res["dP_total"],
            "FBHP":                  res["fbhp"],
        }
