"""
Nodal Analysis Package for Petroleum Engineering
================================================
Implements PIPESIM-grade nodal analysis for production wells.

Modules:
    fluid_properties  - PVT correlations (Standing, Vasquez-Beggs, etc.)
    ipr               - Inflow Performance Relationship models
    pressure_loss     - Pressure gradient calculations
    vlp               - Vertical Lift Performance (tubing correlations)
    completion_design - Packer & completion string design
    nodal_solver      - Operating point solver and analysis orchestrator
"""

from .fluid_properties import FluidProperties
from .ipr import IPRCalculator
from .vlp import VLPCalculator
from .completion_design import CompletionDesigner
from .nodal_solver import NodalAnalysis

__all__ = [
    "FluidProperties",
    "IPRCalculator",
    "VLPCalculator",
    "CompletionDesigner",
    "NodalAnalysis",
]
