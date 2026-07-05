"""
pressure_loss.py
================
Multiphase pressure gradient calculations for vertical/deviated tubing.

Computes the three components of pressure gradient at a point in the tubing:
  1. Gravity (hydrostatic) gradient
  2. Friction gradient
  3. Acceleration (kinetic energy) gradient

Field units:
  Pressure   : psia
  Length/Depth : ft
  Flow rate  : STB/day (surface)
  Velocity   : ft/s
  Density    : lb/ft³
  Viscosity  : cp
  Surface tension : dyne/cm (= mN/m)

References:
  - Brill & Mukherjee, Multiphase Flow in Wells (1999)
  - Beggs & Brill, JPT (1973)
  - Hagedorn & Brown, JPT (1965)
"""

import math
import numpy as np


# Gravitational acceleration in ft/s²
G_C = 32.174      # lbm·ft / (lbf·s²)
G   = 32.174      # ft/s²


def moody_friction_factor(Re: float, roughness: float, D: float) -> float:
    """
    Moody friction factor using the Serghides (1984) explicit approximation
    to the Colebrook-White equation.

    Parameters
    ----------
    Re        : Reynolds number (dimensionless)
    roughness : absolute pipe roughness (ft)
    D         : pipe inner diameter (ft)

    Returns
    -------
    f : Darcy-Weisbach friction factor
    """
    if Re < 1.0:
        return 64.0  # cap for near-zero flow
    if Re < 2000.0:
        return 64.0 / Re   # laminar
    # Turbulent: Serghides (Colebrook-White)
    ed = roughness / D
    A = -2.0 * math.log10(ed / 3.7 + 12.0 / Re)
    B = -2.0 * math.log10(ed / 3.7 + 2.51 / (Re * A))
    C = -2.0 * math.log10(ed / 3.7 + 2.51 / (Re * B))
    f_inv = A - (B - A) ** 2 / (C - 2.0 * B + A)
    f = (1.0 / f_inv) ** 2
    return max(0.005, f)


class PressureLossCalculator:
    """
    Compute pressure loss components along a tubing string for
    multiphase (oil + water + gas) flow.

    Parameters
    ----------
    fluid_props : FluidProperties
        Instance providing mixture PVT properties
    tubing_id : float
        Tubing inner diameter (inches)
    roughness : float
        Absolute pipe wall roughness (ft), default 0.0006 ft
    inclination : float
        Deviation from vertical (degrees). 0 = vertical, 90 = horizontal.
    """

    def __init__(
        self,
        fluid_props,
        tubing_id: float,
        roughness: float = 0.0006,
        inclination: float = 0.0,
    ):
        self.fp = fluid_props
        self.D_in = tubing_id              # inches
        self.D_ft = tubing_id / 12.0       # feet
        self.A = math.pi * self.D_ft ** 2 / 4.0   # ft²
        self.roughness = roughness         # ft
        self.theta = inclination           # degrees from vertical
        self.sin_theta = math.cos(math.radians(inclination))  # sin(from horizontal) = cos(from vertical)

    def _superficial_velocities(self, q_liq_stb: float, P: float, T_F: float):
        """
        Compute superficial velocities for liquid and gas at in-situ conditions.

        Parameters
        ----------
        q_liq_stb : float
            Total liquid surface rate (STB/day)
        P, T_F : float
            Pressure (psia) and temperature (°F)

        Returns
        -------
        v_sl : superficial liquid velocity (ft/s)
        v_sg : superficial gas velocity (ft/s)
        v_m  : mixture velocity (ft/s)
        """
        props = self.fp.liquid_holdup_input(P, T_F)
        ql_ratio = props["ql_res_ratio"]   # RB per STB surface liquid
        qg_ratio = props["qg_res_ratio"]   # RB per STB surface liquid

        # Convert STB/day to ft³/s
        # 1 RB = 5.615 ft³ ; 1 day = 86400 s
        q_L_res = q_liq_stb * ql_ratio * 5.615 / 86400.0   # ft³/s
        q_G_res = q_liq_stb * qg_ratio * 5.615 / 86400.0   # ft³/s

        v_sl = q_L_res / self.A
        v_sg = q_G_res / self.A
        v_m  = v_sl + v_sg
        return v_sl, v_sg, v_m

    # ================================================================== #
    #  BEGGS & BRILL (1973, corrected 1986)                               #
    # ================================================================== #
    def beggs_brill_gradient(
        self, q_liq_stb: float, P: float, T_F: float
    ) -> dict:
        """
        Beggs & Brill multiphase pressure gradient (psi/ft).

        Returns dict with keys:
          gravity_gradient, friction_gradient, accel_gradient, total_gradient
          HL (liquid holdup), rho_mix, flow_regime
        """
        props = self.fp.liquid_holdup_input(P, T_F)
        v_sl, v_sg, v_m = self._superficial_velocities(q_liq_stb, P, T_F)

        rho_L = props["rho_L"]
        rho_g = props["rho_g"]
        mu_L  = props["mu_L"]
        mu_g  = props["mu_g"]
        sigma = props["sigma_L"]
        lambda_L = props["lambda_L"]

        if v_m < 1e-6:
            # No flow: pure hydrostatic
            rho_ns = rho_L * lambda_L + rho_g * (1.0 - lambda_L)
            return {
                "gravity_gradient": rho_ns * self.sin_theta / 144.0,
                "friction_gradient": 0.0,
                "accel_gradient": 0.0,
                "total_gradient": rho_ns * self.sin_theta / 144.0,
                "HL": lambda_L,
                "rho_mix": rho_ns,
                "flow_regime": "No Flow",
            }

        # ---- Flow regime determination --------------------------------
        # Froude number and velocity ratio
        NFr  = v_m ** 2 / (G * self.D_ft)
        NLv  = v_sl * (rho_L / (G * sigma * 6.72e-2)) ** 0.25   # liquid velocity number

        # Transition boundaries (Beggs & Brill, 1986)
        L1 = 316.0 * lambda_L ** 0.302
        L2 = 0.0009252 * lambda_L ** (-2.4684)
        L3 = 0.1 * lambda_L ** (-1.4516)
        L4 = 0.5 * lambda_L ** (-6.738)

        if lambda_L < 0.01 and NFr < L1:
            regime = "Segregated"
        elif lambda_L >= 0.01 and NFr < L2:
            regime = "Segregated"
        elif lambda_L >= 0.01 and NFr > L3 and NFr <= L1:
            regime = "Transition"
        elif lambda_L < 0.4 and NFr >= L1 and NFr <= L4:
            regime = "Intermittent"
        elif lambda_L >= 0.4 and NFr >= L3 and NFr <= L4:
            regime = "Intermittent"
        elif lambda_L < 0.4 and NFr >= L4:
            regime = "Distributed"
        elif lambda_L >= 0.4 and NFr > L4:
            regime = "Distributed"
        else:
            regime = "Segregated"

        # ---- Liquid holdup (horizontal) ----------------------------
        def _HL_horizontal(regime, lambda_L, NFr):
            if regime == "Segregated":
                a, b, c = 0.980, 0.4846, 0.0868
            elif regime == "Intermittent":
                a, b, c = 0.845, 0.5351, 0.0173
            elif regime == "Distributed":
                a, b, c = 1.065, 0.5824, 0.0609
            else:   # Transition – interpolated later
                a, b, c = 0.980, 0.4846, 0.0868
            HL = a * lambda_L ** b / NFr ** c
            return max(lambda_L, min(1.0, HL))

        if regime == "Transition":
            HL_seg = _HL_horizontal("Segregated", lambda_L, NFr)
            HL_int = _HL_horizontal("Intermittent", lambda_L, NFr)
            A_t = (L3 - NFr) / (L3 - L2)
            HL0 = A_t * HL_seg + (1.0 - A_t) * HL_int
        else:
            HL0 = _HL_horizontal(regime, lambda_L, NFr)

        # ---- Inclination correction for upward vertical flow ---------
        # For vertical upward flow, inclination angle from horizontal = 90°
        # beta = 90° for vertical
        if regime == "Segregated":
            e1, e2, e3, f1 = 0.011, -3.768, 3.539, -1.614
        elif regime == "Intermittent":
            e1, e2, e3, f1 = 2.96, 0.305, -0.4473, 0.0978
        else:  # Distributed or Transition
            # No correction for distributed or downhill
            HL = HL0
            e1 = e2 = e3 = f1 = 0.0

        if e1 != 0.0:
            NLv_loc = v_sl * (rho_L / (G * sigma * 6.72e-2)) ** 0.25
            C = max(0.0, (1.0 - lambda_L) * math.log(
                abs(e1) * lambda_L ** e2 * NLv_loc ** e3 * NFr ** f1 + 1e-10
            ))
            psi = 1.0 + C * (math.sin(1.8 * math.radians(90.0))
                              - (1.0/3.0) * math.sin(1.8 * math.radians(90.0)) ** 3)
            HL = HL0 * psi
        else:
            HL = HL0

        HL = max(lambda_L, min(1.0, HL))

        # ---- Mixture properties with holdup --------------------------
        rho_mix = rho_L * HL + rho_g * (1.0 - HL)
        rho_ns  = rho_L * lambda_L + rho_g * (1.0 - lambda_L)

        # ---- Friction factor ------------------------------------------
        mu_m  = mu_L ** lambda_L * mu_g ** (1.0 - lambda_L)
        Re_m  = rho_ns * v_m * self.D_ft / (mu_m * 6.72e-4)   # cp→lb/(ft·s)
        fn    = moody_friction_factor(Re_m, self.roughness, self.D_ft)

        # Holdup correction to friction factor (Beggs & Brill)
        if lambda_L > 0.0 and HL > 0.0:
            y = lambda_L / HL ** 2
            if 1.0 < y < np.inf:
                ln_y = math.log(y)
                s_num = ln_y
                s_den = (-0.0523 + 3.182*ln_y - 0.8725*ln_y**2 + 0.01853*ln_y**4)
                s = s_num / max(abs(s_den), 0.001)
                s = max(-2.0, min(2.0, s))
                f_tp = fn * math.exp(s)
            else:
                f_tp = fn
        else:
            f_tp = fn

        # ---- Pressure gradients (psi/ft) -----------------------------
        # Gravity (hydrostatic)
        dP_grav = rho_mix * self.sin_theta / 144.0

        # Friction  (Darcy-Weisbach)
        dP_fric = f_tp * rho_ns * v_m ** 2 / (2.0 * G_C * self.D_ft * 144.0)

        # Acceleration (kinetic energy)
        # dP_acc = rho_mix × vm × dvm/dL ≈ 0 for incompressible, small for gas
        Ek = rho_ns * v_m * v_sg / (G_C * P * 144.0)  # dimensionless
        if Ek >= 1.0:
            Ek = 0.99
        dP_acc = dP_grav + dP_fric   # placeholder: total/(1-Ek)  later

        total_grad = (dP_grav + dP_fric) / max(1.0 - Ek, 0.01)

        return {
            "gravity_gradient":  dP_grav,
            "friction_gradient": dP_fric,
            "accel_gradient":    total_grad - dP_grav - dP_fric,
            "total_gradient":    total_grad,
            "HL": HL,
            "rho_mix": rho_mix,
            "flow_regime": regime,
        }

    # ================================================================== #
    #  HAGEDORN & BROWN (1965) with Griffith-Wallis bubble correction      #
    # ================================================================== #
    def hagedorn_brown_gradient(
        self, q_liq_stb: float, P: float, T_F: float
    ) -> dict:
        """
        Hagedorn & Brown (1965) correlation.
        Corrected with Griffith-Wallis bubble-flow check.

        Returns same dict structure as beggs_brill_gradient.
        """
        props = self.fp.liquid_holdup_input(P, T_F)
        v_sl, v_sg, v_m = self._superficial_velocities(q_liq_stb, P, T_F)

        rho_L = props["rho_L"]
        rho_g = props["rho_g"]
        mu_L  = props["mu_L"]
        mu_g  = props["mu_g"]
        sigma = props["sigma_L"]
        lambda_L = props["lambda_L"]

        if v_m < 1e-6:
            rho_ns = rho_L * lambda_L + rho_g * (1.0 - lambda_L)
            return {
                "gravity_gradient": rho_ns / 144.0,
                "friction_gradient": 0.0,
                "accel_gradient": 0.0,
                "total_gradient": rho_ns / 144.0,
                "HL": lambda_L,
                "rho_mix": rho_ns,
                "flow_regime": "No Flow",
            }

        # ---- Griffith-Wallis bubble flow check -----------------------
        # If flow is in bubble regime, use simplified approach
        vb = 0.8   # bubble rise velocity (ft/s) – approximate
        if v_sg < 0.25 * v_m and v_sg < vb:
            # Bubble flow: simple mixture approach
            HL_bub = 1.0 - v_sg / (0.8 + v_m)
            HL_bub = max(lambda_L, min(1.0, HL_bub))
            regime = "Bubble"
        else:
            HL_bub = None
            regime = "Slug/Mist"

        if HL_bub is not None:
            HL = HL_bub
        else:
            # ---- CNL correlation dimensionless numbers ---------------
            # Liquid velocity number
            NLv = v_sl * (rho_L / (G * sigma * 6.72e-2)) ** 0.25
            # Gas velocity number
            NGv = v_sg * (rho_L / (G * sigma * 6.72e-2)) ** 0.25
            # Pipe diameter number
            Nd  = self.D_ft * (rho_L * G / (sigma * 6.72e-2)) ** 0.5
            # Liquid viscosity number
            NL  = mu_L * (G / (rho_L * (sigma * 6.72e-2) ** 3)) ** 0.25

            # ---- CNL/Pr holdup correlation (Hagedorn & Brown tables → polynomial fit)
            # Using the commonly cited polynomial approximations
            # CNL correlation: HL/psi
            CNL_coeff = 0.002  # liquid viscosity factor (approx)

            # Pseudo-reduced pressure ratio
            psi_r = (NLv / NGv ** 0.575) * (14.7 / P) ** 0.1 * (CNL_coeff / Nd)
            # Clamp
            psi_r = max(0.01, min(100.0, psi_r))

            # HL polynomial fit (from H&B chart regression)
            # log(HL) vs log(NLv*Pvac^0.1/(NGv^0.575 * Nd * CNL))
            lp = math.log10(psi_r)
            HL = 10 ** (0.816 * lp - 0.0694 * lp**2 + 0.0028 * lp**3)
            HL = max(lambda_L, min(1.0, HL))

        # ---- Mixture density -----------------------------------------
        rho_mix = rho_L * HL + rho_g * (1.0 - HL)
        rho_ns  = rho_L * lambda_L + rho_g * (1.0 - lambda_L)

        # ---- Friction factor (using mixture Reynolds) ----------------
        mu_m  = mu_L ** HL * mu_g ** (1.0 - HL)
        Re_m  = rho_mix * v_m * self.D_ft / (mu_m * 6.72e-4)
        f_tp  = moody_friction_factor(Re_m, self.roughness, self.D_ft)

        # ---- Pressure gradients -------------------------------------
        dP_grav = rho_mix * self.sin_theta / 144.0
        dP_fric = f_tp * rho_mix * v_m ** 2 / (2.0 * G_C * self.D_ft * 144.0)

        # Acceleration (Poettmann & Carpenter term)
        Ek = rho_mix * v_m * v_sg / (G_C * P * 144.0)
        if Ek >= 1.0:
            Ek = 0.99
        total_grad = (dP_grav + dP_fric) / max(1.0 - Ek, 0.01)

        return {
            "gravity_gradient":  dP_grav,
            "friction_gradient": dP_fric,
            "accel_gradient":    total_grad - dP_grav - dP_fric,
            "total_gradient":    total_grad,
            "HL": HL,
            "rho_mix": rho_mix,
            "flow_regime": regime,
        }
