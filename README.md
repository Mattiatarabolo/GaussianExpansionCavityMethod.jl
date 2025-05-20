# GaussianExpansionCavityMethod

*Gaussian Expansion of the Dynamic Cavity method for coupled Stochastic Differential Equations*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mattiatarabolo.github.io/GaussianExpansionCavityMethod.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Mattiatarabolo.github.io/GaussianExpansionCavityMethod.jl/dev/)
[![Build Status](https://github.com/Mattiatarabolo/GaussianExpansionCavityMethod.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Mattiatarabolo/GaussianExpansionCavityMethod.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Mattiatarabolo/GaussianExpansionCavityMethod.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattiatarabolo/GaussianExpansionCavityMethod.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


# Overview
The purpose of _GaussianExpansionCavityMethod.jl_ is to provide a Julia implementation of the Gaussian Expansion Cavity Method (GECaM) for coupled Stochastic Differential Equations (SDEs). This method is used to study the dynamics of systems with many interacting components, such as spin glasses, neural networks, and other complex systems.

The package offers many functions to simulate the dynamics of coupled SDEs through Monte Carlo (MC) numerical integration, as well as numerical implementations of the GECaM equations. In particular, it has been applied to the following models:
- Linearly coupled Ornstein-Uhlenbeck (OU) processes.
- Phi4 model, i.e. linearly coupled OU processes with a quartic potential.
- Bouchaud-Mezard model on sparse graphs.
- Spherical 2-spin model on sparse graphs.

More details can be found in the [examples](./examples/) folder.

# Installation

The package can be installed using the Julia package manager. Open Julia and type:
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia-repl
pkg> add https://github.com/Mattiatarabolo/GaussianExpansionCavityMethod.jl
```

Or, equivalently, via the `Pkg` API:

```julia-repl
julia> import Pkg; Pkg.add(url="https://github.com/Mattiatarabolo/GaussianExpansionCavityMethod.jl")
```

# References

The package is based on the following paper:
- [M. Tarabolo and L. Dall'Asta, Gaussian approximation of dynamic cavity equations for linearly-coupled stochastic dynamics, SciPost Physics (2025). Still under review.](https://scipost.org/submissions/scipost_202502_00024v2/) 