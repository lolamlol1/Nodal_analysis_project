"""
fluid_properties.py
===================
PVT correlations for reservoir fluid characterization.

All correlations use field units:
  Pressure    : psia
  Temperature : °F (converted to °R internally where needed)
  GOR         : scf/STB
  Bo          : RB/STB
  Viscosity   : cp
  Density     : lb/ft³

References:
  - Standing (1947/1977) – bubble-point, Bo, GOR
  - Beggs & Robinson (1975) – dead-oil viscosity
  - Chew & Connally (1959) – live-oil viscosity
  - Hall & Yarborough (1974) – gas z-factor
  - Lee, Gonzalez & Eakin (1966) – gas viscosity
  - Brill & Mukherjee (1999) – water properties
"""

import math
import numpy as np


class FluidProperties:
    """
    Compute PVT properties for an oil-gas-water system.

    Parameters
    ----------
    api_gravity : float
        Stock-tank oil API gravity (°API)
    gas_sg : float
        Gas specific gravity (air = 1.0)
    water_sg : float
        Water specific gravity (fresh water = 1.0)
    gor : float
        Producing GOR at standard conditions (scf/STB)
    water_cut : float
        Water cut as a fraction (0.0 – 1.0), e.g. 0.20 for 20%
    reservoir_temp : float
        Reservoir temperature (°F)
    """

    def __init__(
        self,
        api_gravity: float = 35.0,
        gas_sg: float = 0.65,
        water_sg: float = 1.07,
        gor: float = 500.0,
        water_cut: float = 0.20,
        reservoir_temp: float = 180.0,
    ):
        self.api = api_gravity
        self.gas_sg = gas_sg
        self.water_sg = water_sg
        self.gor = gor          # total GOR (scf/STB of oil)
        self.wc = water_cut     # fraction
        self.T_res = reservoir_temp  # °F

        # Derived base properties
        self.oil_sg = 141.5 / (self.api + 131.5)
        self.rho_oil_sc = self.oil_sg * 62.4    # lb/ft³ at SC
        self.rho_water_sc = self.water_sg * 62.4
        self.rho_gas_sc = self.gas_sg * 0.0764  # lb/ft³ at SC (air ~ 0.0764 lb/ft³)

        # Bubble-point pressure (at reservoir temperature)
        self.bubble_point = self._bubble_point_standing(self.T_res, self.gor)

    # ------------------------------------------------------------------ #
    #  BUBBLE-POINT PRESSURE  (Standing, 1947)                            #
    # ------------------------------------------------------------------ #
    def _bubble_point_standing(self, T_F: float, Rs: float) -> float:
        """
        Standing (1947) bubble-point pressure correlation.
        T_F  : temperature (°F)
        Rs   : solution GOR at bubble point (scf/STB)
        Returns Pb in psia.
        """
        if Rs <= 0:
            return 14.7
        x = 0.0125 * self.api - 0.00091 * T_F
        Pb = 18.2 * ((Rs / self.gas_sg) ** 0.83 * (10.0 ** x) - 1.4)
        return max(14.7, Pb)

    # ------------------------------------------------------------------ #
    #  SOLUTION GOR  (Standing, 1977 – inverse of bubble-point)           #
    # ------------------------------------------------------------------ #
    def solution_gor(self, P: float, T_F: float) -> float:
        """
        Solution GOR at pressure P and temperature T_F (Standing).
        Returns Rs in scf/STB.  Rs <= producing GOR always.
        """
        if P >= self.bubble_point:
            return self.gor  # all gas in solution above bubble-point
        # Standing inverse: Rs = gamma_g * ((Pb/18.2 + 1.4) / 10^x)^(1/0.83)
        x = 0.0125 * self.api - 0.00091 * T_F
        Rs = self.gas_sg * ((P / 18.2 + 1.4) / (10.0 ** x)) ** (1.0 / 0.83)
        return max(0.0, min(Rs, self.gor))

    # ------------------------------------------------------------------ #
    #  OIL FORMATION VOLUME FACTOR  (Standing, 1947)                      #
    # ------------------------------------------------------------------ #
    def oil_fvf(self, P: float, T_F: float) -> float:
        """Oil FVF Bo (RB/STB) – Standing (1947)."""
        Rs = self.solution_gor(P, T_F)
        F = Rs * (self.gas_sg / self.oil_sg) ** 0.5 + 1.25 * T_F
        Bo = 0.972 + 1.47e-4 * F ** 1.175
        return max(1.0, Bo)

    # ------------------------------------------------------------------ #
    #  GAS Z-FACTOR  (Papay, 1968 – fast explicit approximation)          #
    # ------------------------------------------------------------------ #
    def gas_z_factor(self, P: float, T_F: float) -> float:
        """
        Papay (1968) explicit z-factor correlation – fast, accurate for
        typical natural gas conditions (Tpr > 1.05, Ppr < 15).
        Sutton (1985) pseudo-critical properties.
        """
        # Sutton pseudo-critical properties
        Tpc = 169.2 + 349.5 * self.gas_sg - 74.0 * self.gas_sg ** 2  # °R
        Ppc = 756.8 - 131.0 * self.gas_sg - 3.6 * self.gas_sg ** 2   # psia

        T_R  = T_F + 459.67
        Tpr  = T_R / Tpc
        Ppr  = P   / Ppc

        if Ppr < 0.001:
            return 1.0

        # Papay (1968) – explicit approximation
        z = (1.0
             - 3.52 * Ppr / (10.0 ** (0.9813 * Tpr))
             + 0.274 * Ppr ** 2 / (10.0 ** (0.8157 * Tpr)))

        # Clamp to physically valid range
        return max(0.15, min(2.5, z))

    # ------------------------------------------------------------------ #
    #  GAS FORMATION VOLUME FACTOR                                         #
    # ------------------------------------------------------------------ #
    def gas_fvf(self, P: float, T_F: float) -> float:
        """Bg in ft³/SCF at reservoir conditions."""
        z = self.gas_z_factor(P, T_F)
        T_R = T_F + 459.67
        Bg = 0.02829 * z * T_R / max(P, 14.7)  # ft³/SCF
        return max(1e-6, Bg)

    # ------------------------------------------------------------------ #
    #  OIL VISCOSITY  (Beggs & Robinson 1975 dead; Chew & Connally 1959)  #
    # ------------------------------------------------------------------ #
    def oil_viscosity(self, P: float, T_F: float) -> float:
        """Live-oil viscosity in cp."""
        # Dead-oil viscosity (Beggs & Robinson, 1975)
        x = 10.0 ** (3.0324 - 0.02023 * self.api) / (T_F ** 1.163)
        mu_od = max(0.01, 10.0 ** x - 1.0)

        Rs = self.solution_gor(P, T_F)

        # Chew & Connally (1959) live-oil viscosity
        A = 10.715 * (Rs + 100.0) ** (-0.515)
        B = 5.44 * (Rs + 150.0) ** (-0.338)
        mu_o = A * mu_od ** B
        return max(0.1, mu_o)

    # ------------------------------------------------------------------ #
    #  GAS VISCOSITY  (Lee, Gonzalez & Eakin, 1966)                       #
    # ------------------------------------------------------------------ #
    def gas_viscosity(self, P: float, T_F: float) -> float:
        """Gas viscosity in cp."""
        T_R = T_F + 459.67
        Mg = 28.97 * self.gas_sg  # molecular weight g/mol
        z = self.gas_z_factor(P, T_F)
        rho_g = P * Mg / (z * 10.73 * T_R)  # lb/ft³

        K = ((9.4 + 0.02 * Mg) * T_R ** 1.5) / (209.0 + 19.0 * Mg + T_R)
        X = 3.5 + 986.0 / T_R + 0.01 * Mg
        Y = 2.4 - 0.2 * X
        mu_g = 1e-4 * K * math.exp(X * (rho_g / 62.4) ** Y)
        return max(0.005, mu_g)

    # ------------------------------------------------------------------ #
    #  WATER VISCOSITY & FVF                                               #
    # ------------------------------------------------------------------ #
    def water_viscosity(self, T_F: float) -> float:
        """Water viscosity in cp (Van Wingen, 1950 approximation)."""
        mu_w = math.exp(1.003 - 1.479e-2 * T_F + 1.982e-5 * T_F ** 2)
        return max(0.1, mu_w)

    def water_fvf(self, P: float, T_F: float) -> float:
        """Water FVF Bw in RB/STB (Craft & Hawkins)."""
        Bw = (
            1.0
            + 1.21e-4 * (T_F - 60.0)
            + 1.0e-6 * (T_F - 60.0) ** 2
            - 3.33e-6 * P
        )
        return max(1.0, Bw)

    # ------------------------------------------------------------------ #
    #  MIXTURE PROPERTIES  (for VLP correlations)                          #
    # ------------------------------------------------------------------ #
    def liquid_holdup_input(self, P: float, T_F: float) -> dict:
        """
        Compute all mixture properties needed by the VLP correlations.

        Returns a dict with:
          rho_L          : in-situ liquid density (lb/ft³)
          rho_g          : in-situ gas density (lb/ft³)
          mu_L           : liquid viscosity (cp)
          mu_g           : gas viscosity (cp)
          sigma_L        : liquid surface tension (dyne/cm)
          lambda_L       : no-slip liquid holdup (volume fraction)
          ql_res_ratio   : liquid volume at res conditions per STB surface liquid (RB/STB)
          qg_res_ratio   : gas volume at res conditions per STB surface liquid (RB/STB)
        """
        Rs = self.solution_gor(P, T_F)
        Bo = self.oil_fvf(P, T_F)
        Bw = self.water_fvf(P, T_F)
        Bg = self.gas_fvf(P, T_F)   # ft³/SCF

        oil_frac   = 1.0 - self.wc
        water_frac = self.wc
        free_gor   = max(0.0, self.gor - Rs)  # scf/STB oil (free gas)

        # Volumetric rates at reservoir conditions per STB total surface liquid
        ql_oil   = oil_frac * Bo           # RB/STB
        ql_water = water_frac * Bw         # RB/STB
        ql_res   = ql_oil + ql_water       # RB/STB (total liquid)
        # Gas volume: oil_frac × free_gor [scf/STB oil] × Bg [ft³/scf] / 5.615 [ft³/RB]
        qg_res   = oil_frac * free_gor * Bg / 5.615  # RB/STB

        total_res = ql_res + qg_res
        lambda_L  = ql_res / max(total_res, 1e-10)

        # Phase densities at reservoir conditions
        # Oil: mass of oil + dissolved gas, divided by res volume
        rho_oil_res = (self.rho_oil_sc * oil_frac
                       + self.rho_gas_sc * Rs * oil_frac / 5.615) / max(ql_oil, 1e-10)
        rho_water_res = self.rho_water_sc / Bw
        if ql_res > 1e-10:
            rho_L = (ql_oil * rho_oil_res + ql_water * rho_water_res) / ql_res
        else:
            rho_L = self.rho_oil_sc

        # Gas density at reservoir conditions
        rho_g = max(0.01, self.rho_gas_sc / max(Bg * 5.615, 1e-6))  # lb/ft³

        # Viscosities
        mu_oil   = self.oil_viscosity(P, T_F)
        mu_water = self.water_viscosity(T_F)
        mu_gas   = self.gas_viscosity(P, T_F)
        if ql_res > 1e-10:
            mu_L = (ql_oil * mu_oil + ql_water * mu_water) / ql_res
        else:
            mu_L = mu_oil

        # Surface tension (dyne/cm ≈ mN/m)
        sigma_oil   = max(1.0, 68.0 - 0.000015 * P - 0.5 * self.api)
        sigma_water = max(1.0, 74.0 - 0.1 * (T_F - 60.0))
        sigma_L     = oil_frac * sigma_oil + water_frac * sigma_water

        return {
            "rho_L":         max(5.0, rho_L),
            "rho_g":         max(0.01, rho_g),
            "mu_L":          max(0.05, mu_L),
            "mu_g":          max(0.005, mu_gas),
            "sigma_L":       max(1.0, sigma_L),
            "lambda_L":      max(0.001, min(1.0, lambda_L)),
            "ql_res_ratio":  max(1e-6, ql_res),
            "qg_res_ratio":  max(0.0, qg_res),
        }

    def mixture_density(self, P: float, T_F: float) -> float:
        """Mixture density (oil + water + free gas) in lb/ft³ at P, T."""
        props = self.liquid_holdup_input(P, T_F)
        lam   = props["lambda_L"]
        return lam * props["rho_L"] + (1.0 - lam) * props["rho_g"]
