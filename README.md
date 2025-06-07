# Accelerating Phase Diagram Construction through Activity Coefficient Prediction

This repository contains all code and data supporting the paper:

## 🧠 Overview

This repository, alongside the main text, introduces a machine learning workfolwo to reconstruct vapor–liquid phase diagrams by training a Gaussian Process Regression (GPR) model to map **Kirkwood–Buff integrals (KBIs)** to **activity coefficients**.

Instead of expensive two-phase simulations, we compute KBIs from single-phase simulations and use the trained GP model to predict activity coefficients, which are then used to reconstruct phase diagrams.

### 📈 Method Flow:

1. Compute RDFs from single-phase MD
2. Integrate to get KBIs \( G_{11}, G_{12}, G_{22} \)
3. Train GPR on known systems: \( [G_{ij}] \to \gamma_1, \gamma_2 \)
4. Predict \( \gamma_i \) for new systems
5. Reconstruct phase diagrams using \( \gamma_i \) and saturation pressures

## 📁 Files

| example codes |
|------|-------------|
| `data.sh` | Example of input data generation for MD simulation in LAMMPS |
| `KBI.py` | Reads RDFs and computes corrected KBIs |
| `PhaseDiagram.py` | MD construction of VLE phase diagrams using pressure and density profiles from MD trajectories with four subplots of densities, pressure, and resulting pphase diagram shown in SI, Figures S1 |
| `PhaseDiagram-train.py` | MD construction of VLE phase diagrams using pressure and density profiles from MD trajectories with four subplots of densities, pressure, and resulting pphase diagram shown in SI, Figures S1 |
| `GP.py` | GP model |
| `kbi_data.csv` | Tabulated \( G_{ij} \) values used as GP input |
| `activity_coeffs.csv` | MD-extracted \( \gamma_1, \gamma_2 \) values for GP output |
