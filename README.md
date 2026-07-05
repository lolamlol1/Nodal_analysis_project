# 🛢️ Nodal Analysis Suite

**PIPESIM-grade Nodal Analysis** for production wells — built with Python & Streamlit.

## Features
- **5 IPR Models**: Vogel, Linear PI, Fetkovitch, Standing, Jones Composite  
- **2 VLP Correlations**: Beggs & Brill (1973), Hagedorn & Brown (1965)  
- **Full PVT Engine**: Standing, Papay z-factor, Beggs-Robinson viscosity, Lee-Gonzalez gas viscosity  
- **Animated Charts**: Curves draw live when you open each tab  
- **Completion Design**: Packer depth, tubing string schematic  
- **Sensitivity Analysis**: WHP, Water Cut, GOR sweeps  
- **Downloadable Report**: Full markdown report of every run  

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Streamlit web application
├── requirements.txt
└── nodal/
    ├── fluid_properties.py   # PVT correlations
    ├── ipr.py                # Inflow Performance Relationship models
    ├── pressure_loss.py      # Multiphase gradient calculations
    ├── vlp.py                # Vertical Lift Performance curves
    ├── completion_design.py  # Completion string & packer design
    └── nodal_solver.py       # Operating point solver (IPR ∩ VLP)
```

## Physics References
- Standing (1947) — Bubble-point, oil FVF, solution GOR  
- Beggs & Robinson (1975) — Dead-oil viscosity  
- Chew & Connally (1959) — Live-oil viscosity  
- Papay (1968) — Gas z-factor  
- Lee, Gonzalez & Eakin (1966) — Gas viscosity  
- Beggs & Brill (1973) — Multiphase flow in pipes  
- Hagedorn & Brown (1965) — Vertical multiphase flow  
- Vogel (1968) — IPR for solution-gas-drive reservoirs  
