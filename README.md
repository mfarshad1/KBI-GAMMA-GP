# Accelerating Phase Diagram Construction through Activity Coefficient Prediction

This repository contains all code and data supporting the paper:

**"Accelerating Phase Diagram Construction through Activity Coefficient Prediction"**  
Mohsen Farshad, Fathya Y. M. Salih, Dinis O. Abranches, Yamil J. Colón

## 🧠 Overview

This repository, alongside the main text, introduces a machine learning workflow to reconstruct vapor–liquid equilibrium (VLE) phase diagrams by training a Gaussian Process Regression (GPR) model to map **Kirkwood–Buff integrals (KBIs)** to **activity coefficients**.

Instead of relying on computationally expensive two-phase simulations, we compute KBIs from single-phase molecular dynamics (MD) trajectories and use the trained GP model to predict activity coefficients. These are then used to reconstruct phase diagrams.

### 📈 Method Flow:

1. Compute RDFs from single-phase MD simulations
2. Integrate RDFs to obtain KBIs: \( G_{11}, G_{12}, G_{22} \)
3. Train GPR model using known systems: \( [G_{ij}] \to \gamma_1, \gamma_2 \)
4. Predict \( \gamma_i \) for new systems from KBIs
5. Reconstruct VLE phase diagrams using predicted \( \gamma_i \) and known pure-component saturation pressures

## 📁 Files

| File | Description |
|------|-------------|
| `data-single-phase.sh` | Example script to generate LAMMPS input configurations for single-phase binary LJ systems |
| `data-biphase.sh` | Example script to generate LAMMPS input configurations for biphasic binary LJ systems |
| `md.in` | LAMMPS input file for either single-phase or biphasic binary Lennard-Jones mixture simulations.
| `md-pure1.in` | LAMMPS input file for either single-phase or biphasic pure component 1 simulations.
| `md-pure2.in` | LAMMPS input file for either single-phase or biphasic pure component 2 simulations.
| `batch` | Slurm batch script to run MD simulations for either single-phase or biphasic binary Lennard-Jones mixtures.
| `batch-pure1` | Slurm batch script to run MD simulations for either single-phase or biphasic pure component 1.
| `batch-pure2` | Slurm batch script to run MD simulations for either single-phase or biphasic pure component 2.
| `KBI.py` | Computes RDF, KBIs, corrected KBIs for a given system extracted from MD single-phase simulation. See SI Figures S3–S12 for system-specific results |
| `KBI-all.py` | Computes corrected Kirkwood–Buff integrals for all systems of single-phase simulations. See Figure 1.  |
| `PhaseDiagram.py` | Constructs VLE phase diagrams from MD pressure and density profiles extracted from MD biphasic simulations. Outputs shown in SI Figure S1 |
| `PhaseDiagram-train.py` | Constructs phase diagrams for all three training systems extracted from MD biphasic simulations. Final plots shown in SI Figure S2 |
| `PhaseDiagram-test.py` | Compares ML-predicted phase diagrams against MD for test systems. See main text Figure 4 |
| `GP.py` | Main script for training and evaluating the Gaussian Process model. See Figures 2 and 3. |
| `PhaseDiagram.xlsx` | An Excel file that contains tabulated KBIs (SI, Tables S2-S6), activity coefficients for training (Table S1) and test systems (Table S7), predicted activity coefficients for test systems (Table S7), and the corresponding reconstructed phase diagrams. |
