# AIROBAS: AI ROBustness ASsessment

## Introduction

**What is AIROBAS?** It is a library that allows the combinaison of verification tools for neural network into a pipeline.

It provides functionalities to:

– load input data and models,

– define a robustness property to be verified. The code has been implemented
to be as generic as possible and highly parametrizable regarding property
definition,

– design a customized verification pipeline,

– run verification with a default set of empirical and formal methods e.g., adversarial attacks using the [cleverhans](https://github.com/cleverhans-lab/cleverhans) library, incomplete verification using LiRPA-based
functionalities implemented in the Airbus open-source library [decomon](https://github.com/airbus/decomon) etc.,

– append new verification functions or link the code to the open-source libraries of one’s choice

AIROBAS is a complementary tool to existing libraries for the robustness verification of neural networks.
This open source code is a complement to the publication ["Surrogate Neural Networks Local Stability for
Aircraft Predictive Maintenance"](https://arxiv.org/abs/2401.06821) which was peer-reviewed and accepted at the 29th International Conference on Formal Methods for Industrial Critical Systems ([FMICS2024](https://fmics.inria.fr/2024/)).

**Complete Verification Pipeline for Stability**

Verification techniques for neural networks take test points, a trained model and a given property (robustness, stability, monotonicity etc.) as input and return for every test point an assessment of whether or not the property is violated or verified.

The current state-of-the-art mainly encompasses three families of methods:

- The ’no/maybe’ methods (Family of techniques 'A' in paper): e.g. [adversarial attacks](https://github.com/cleverhans-lab/cleverhans). These methods essentially rely on the search for counterexamples that violate the targeted property. If no counterexample is found, no conclusion can be drawn on the model output w.r.t to the targeted property.

- The ’yes/maybe’ methods (Family of techniques 'B' in paper): e.g. the affine bounds generation methods implemented in the [decomon library](https://github.com/airbus/decomon). These techniques intend to bound the output values of a network. If the derived bounds respect the property, it is verified. If the bounds exceed it, no conclusion can be drawn. It can then either mean that the property is violated or that the derived bounds are too loose. These methods are commonly refered to as 'incomplete' (since they can guarantee the property is respected but not that it is violated). They are however generally faster than 'complete' methods (see next).

- The ’yes/no’ methods (Family of techniques 'C' in paper): e.g., [SMT](https://github.com/NeuralNetworkVerification/Marabou), [MILP](https://gurobi-machinelearning.readthedocs.io/en/stable/index.html) solvers etc. They can guarantee that the neural network output respect or not the targeted property, at the cost of significant computation time. They are commonly refered to as 'complete' techniques.

Here is an example of NN verification pipeline:

<div align="center">
    <img src="https://github.com/airbus/Airobas/blob/main/docs/pipeline.png" width="75%" alt="pipeline" align="center" />
</div>


## Installation

Quick version:
```shell
pip install .
```

## Examples

Notebooks providing examples of the use of the library are available in the "tutorials" folder:

- Use case "Aircraft braking distance estimation" [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/airbus/Airobas/blob/main/tutorials/braking_distance_estimation_verification.ipynb)

- Use case "NN surrogate for Rosenbrock function" [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/airbus/Airobas/blob/main/tutorials/rosenbrock_verification.ipynb)
