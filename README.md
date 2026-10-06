# Neural Network Partitioning of NEE in Tidal Marsh Ecosystems

Extending a published neural network method for partitioning carbon fluxes to tidal marshes, and testing whether it improves on standard partitioning methods there.

## Background

Eddy covariance towers measure **Net Ecosystem Exchange (NEE)**, the net CO₂ flux between an ecosystem and the atmosphere. NEE is the balance of two gross fluxes that can't be measured directly:

- **GPP** (gross primary production): carbon taken up by photosynthesis
- **RECO** (ecosystem respiration): carbon released by all respiration processes

Separating NEE into GPP and RECO ("partitioning") is essential for understanding how ecosystems respond to climate. The methods used across FLUXNET rely on physiological relationships with a small set of drivers (air temperature, vapor pressure deficit, radiation), so they can miss the multiple co-acting factors that shape these fluxes.

[Tramontana et al. (2020)](https://doi.org/10.1111/gcb.15203) proposed **NN<sub>C-part</sub>**, a hybrid data-driven approach based on combined neural networks, and validated it on terrestrial FLUXNET sites.

## What this project does

Tidal marshes are an important but under-studied carbon sink. Tidal flooding and salinity add drivers that standard partitioning methods were never designed for. This project:

1. **Extends the validation of NN<sub>C-part</sub> to tidal marsh ecosystems**, comparing its GPP and RECO estimates against traditional partitioning methods.
2. **Quantifies the impact of environmental predictors** on model accuracy, testing which drivers matter in a tidal setting.
3. **Compares model performance rigorously**, using bootstrap confidence intervals and statistical hypothesis testing so that performance differences are reported with uncertainty rather than as single point estimates.

