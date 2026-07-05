"""
completion_design.py
====================
Packer and completion string design for oil wells.

Implements a PIPESIM-style completion string design algorithm:
  - Packer depth selection (based on perforation top + safety margin)
  - SCSSV (Surface-Controlled Subsurface Safety Valve) placement
  - Landing nipple placement
  - Seal assembly placement
  - Multiple tubing size candidate evaluation

Field units: ft (depth), psia (pressure)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .vlp import STANDARD_TUBING_SIZES


@dataclass
class PackerResult:
    """Result of packer depth calculation."""
    packer_depth: float          # ft
    tubing_length: float         # ft (surface to packer)
    clearance: float             # ft (perf top to packer)
    status: str                  # 'VALID' or 'INVALID'
    message: str = ""


@dataclass
class CompletionString:
    """Full completion string design."""
    packer_depth: float          # ft
    tubing_length: float         # ft
    scssv_depth: float           # ft
    landing_nipple_depth: float  # ft
    seal_assembly_depth: float   # ft
    bottom_of_tubing: float      # ft
    tubing_name: str             # e.g. "2-7/8\""
    tubing_id: float             # inches
    status: str                  # 'VALID' or 'INVALID'
    operating_rate: float = 0.0  # STB/day at operating point
    fbhp: float = 0.0            # psia at operating point


@dataclass
class DesignCandidate:
    """A candidate completion design with evaluation results."""
    tubing_name: str
    tubing_id: float             # inches
    packer_depth: float          # ft
    tubing_length: float         # ft
    roughness: float = 0.0006    # ft
    evaluated: bool = False
    operating_rate: float = 0.0
    fbhp: float = 0.0
    score: float = 0.0           # higher = better
    notes: str = ""


class CompletionDesigner:
    """
    Design and evaluate completion strings for a production well.

    Parameters
    ----------
    well_tvd : float
        True vertical depth of well (ft)
    well_md : float
        Measured depth of well (ft)
    perf_top : float
        Top of perforation interval (ft MD)
    perf_bottom : float
        Bottom of perforation interval (ft MD)
    safety_margin : float
        Minimum clearance between packer and top perforation (ft), default 120
    scssv_offset : float
        SCSSV placement above packer (ft), default 300
    landing_nipple_offset : float
        Landing nipple above packer (ft), default 80
    seal_assembly_offset : float
        Seal assembly above packer (ft), default 5
    """

    def __init__(
        self,
        well_tvd: float,
        well_md: float,
        perf_top: float,
        perf_bottom: float,
        safety_margin: float = 120.0,
        scssv_offset: float = 300.0,
        landing_nipple_offset: float = 80.0,
        seal_assembly_offset: float = 5.0,
    ):
        self.tvd = well_tvd
        self.md = well_md
        self.perf_top = perf_top
        self.perf_bottom = perf_bottom
        self.perf_mid = (perf_top + perf_bottom) / 2.0
        self.safety_margin = safety_margin
        self.scssv_offset = scssv_offset
        self.ln_offset = landing_nipple_offset
        self.seal_offset = seal_assembly_offset

    # ------------------------------------------------------------------ #
    #  PACKER DEPTH                                                         #
    # ------------------------------------------------------------------ #
    def calculate_packer_depth(
        self,
        custom_depth: Optional[float] = None,
    ) -> PackerResult:
        """
        Calculate the recommended packer setting depth.

        Parameters
        ----------
        custom_depth : float, optional
            Override depth (ft). If None, computed automatically.

        Returns
        -------
        PackerResult
        """
        if custom_depth is not None:
            packer_depth = custom_depth
        else:
            packer_depth = self.perf_top - self.safety_margin

        clearance = self.perf_top - packer_depth

        if packer_depth <= 0:
            return PackerResult(
                packer_depth=packer_depth,
                tubing_length=max(0.0, packer_depth),
                clearance=clearance,
                status="INVALID",
                message="Packer depth must be positive.",
            )
        if clearance < 0:
            return PackerResult(
                packer_depth=packer_depth,
                tubing_length=packer_depth,
                clearance=clearance,
                status="INVALID",
                message="Packer depth is below top perforation.",
            )
        if clearance < 50:
            status = "WARNING"
            msg = f"Clearance {clearance:.0f} ft is very small (< 50 ft)."
        else:
            status = "VALID"
            msg = f"Recommended packer depth: {packer_depth:.0f} ft with {clearance:.0f} ft clearance."

        return PackerResult(
            packer_depth=packer_depth,
            tubing_length=packer_depth,
            clearance=clearance,
            status=status,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    #  FULL COMPLETION STRING DESIGN                                        #
    # ------------------------------------------------------------------ #
    def design_string(
        self,
        tubing_name: str,
        tubing_id: float,
        packer_depth: Optional[float] = None,
    ) -> CompletionString:
        """
        Design a complete completion string.

        Parameters
        ----------
        tubing_name : str
            API tubing size label
        tubing_id : float
            Tubing inner diameter (inches)
        packer_depth : float, optional
            Override packer depth (ft)

        Returns
        -------
        CompletionString
        """
        packer = self.calculate_packer_depth(packer_depth)

        pd = packer.packer_depth
        scssv_depth = pd - self.scssv_offset
        landing_depth = pd - self.ln_offset
        seal_depth = pd - self.seal_offset

        status = packer.status
        if scssv_depth <= 0:
            status = "INVALID"

        return CompletionString(
            packer_depth=pd,
            tubing_length=pd,
            scssv_depth=max(0.0, scssv_depth),
            landing_nipple_depth=max(0.0, landing_depth),
            seal_assembly_depth=max(0.0, seal_depth),
            bottom_of_tubing=pd,
            tubing_name=tubing_name,
            tubing_id=tubing_id,
            status=status,
        )

    # ------------------------------------------------------------------ #
    #  CANDIDATE GENERATION & RANKING                                       #
    # ------------------------------------------------------------------ #
    def generate_candidates(
        self,
        packer_range_above_perf: tuple = (50, 500),
        packer_step: float = 50.0,
        tubing_sizes: Optional[dict] = None,
    ) -> List[DesignCandidate]:
        """
        Generate all candidate designs by sweeping packer depths and tubing sizes.

        Parameters
        ----------
        packer_range_above_perf : (min, max) ft above perforation top
        packer_step : depth step (ft)
        tubing_sizes : dict {name: id_inches}; defaults to standard API sizes

        Returns
        -------
        List[DesignCandidate]
        """
        if tubing_sizes is None:
            tubing_sizes = STANDARD_TUBING_SIZES

        min_clear, max_clear = packer_range_above_perf
        start = self.perf_top - max_clear
        stop  = self.perf_top - min_clear

        depths = np.arange(
            max(100.0, start),
            stop + packer_step,
            packer_step
        )

        candidates = []
        for depth in depths:
            for name, id_in in tubing_sizes.items():
                c = DesignCandidate(
                    tubing_name=name,
                    tubing_id=id_in,
                    packer_depth=float(depth),
                    tubing_length=float(depth),
                )
                candidates.append(c)
        return candidates

    def rank_candidates(
        self,
        candidates: List[DesignCandidate],
        nodal_solver,
        ipr_calc,
        fluid_props,
        wellhead_pressure: float = 100.0,
        reservoir_temp: float = 180.0,
        ipr_model: str = "vogel",
        vlp_correlation: str = "beggs_brill",
    ) -> List[DesignCandidate]:
        """
        Evaluate and rank all candidates by operating flow rate.

        Parameters
        ----------
        candidates : list of DesignCandidate
        nodal_solver : NodalAnalysis instance factory function / class
        ipr_calc : IPRCalculator instance
        fluid_props : FluidProperties instance
        wellhead_pressure : float, psia
        reservoir_temp : float, °F
        ipr_model : str
        vlp_correlation : str

        Returns
        -------
        List[DesignCandidate] sorted by operating_rate descending
        """
        from .vlp import VLPCalculator
        from .nodal_solver import NodalAnalysis

        for c in candidates:
            try:
                vlp = VLPCalculator(
                    fluid_props=fluid_props,
                    tvd=self.tvd,
                    md=c.packer_depth,
                    tubing_id=c.tubing_id,
                    wellhead_pressure=wellhead_pressure,
                    reservoir_temp=reservoir_temp,
                    roughness=c.roughness,
                )
                solver = NodalAnalysis(
                    ipr_calc=ipr_calc,
                    vlp_calc=vlp,
                )
                op = solver.operating_point(
                    ipr_model=ipr_model,
                    vlp_correlation=vlp_correlation,
                )
                c.operating_rate = op["q"]
                c.fbhp = op["Pwf"]
                c.score = op["q"]
                c.evaluated = True
                c.notes = f"q={op['q']:.0f} STB/d, Pwf={op['Pwf']:.0f} psi"
            except Exception as e:
                c.evaluated = False
                c.score = 0.0
                c.notes = f"Error: {e}"

        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def summary_table(self, design: CompletionString) -> dict:
        """
        Return a formatted summary dict for the designed completion string.
        """
        return {
            "Tubing Size": design.tubing_name,
            "Tubing ID (in)": f"{design.tubing_id:.3f}",
            "Packer Depth (ft)": f"{design.packer_depth:.1f}",
            "Tubing Length (ft)": f"{design.tubing_length:.1f}",
            "SCSSV Depth (ft)": f"{design.scssv_depth:.1f}",
            "Landing Nipple Depth (ft)": f"{design.landing_nipple_depth:.1f}",
            "Seal Assembly Depth (ft)": f"{design.seal_assembly_depth:.1f}",
            "Bottom of Tubing (ft)": f"{design.bottom_of_tubing:.1f}",
            "Perforation Top (ft)": f"{self.perf_top:.1f}",
            "Perforation Bottom (ft)": f"{self.perf_bottom:.1f}",
            "Clearance Packer→Perf (ft)": f"{self.perf_top - design.packer_depth:.1f}",
            "Design Status": design.status,
        }
