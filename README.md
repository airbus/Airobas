# AIROBAS: AI ROBustness Assessment

## Introduction

**What is Airobas?** `A(i)robas` is a library that combines verification tool for neural network into a pipeline.
It provides functionalities to:

– load input data and models,

– define a robustness property to be verified. The code has been implemented
to be as generic as possible and highly parametrizable regarding property
definition,

– design customized verification pipeline,

– run verification with a default set of empirical and formal methods e.g., at-
tacks using the cleverhans library, incomplete verification using LiRPA-based
functionalities implemented in [decomon](https://github.com/airbus/decomon) etc.,

– append new verification functions or link the code to open-source libraries
of one’s choice

We believe that Airobas is a complementary tool to existing libraries for the certification of neural networks.
This open source code is a support to our publication ["Surrogate Neural Networks Local Stability for
Aircraft Predictive Maintenance"](https://arxiv.org/abs/2401.06821)

**Complete Verification Pipeline for Stability**
The state-of-the-art property verification of neural networks is currently divided into three
families of methods. They all take as input test points and a trained model and return
for every test point if the property is violated or verified.
- Firstly, there are the ’no/maybe’ methods (block A), such as adversarial attacks Essentially, these methods rely on the search for counterexamples that would
contradict the stability property. If no counterexample is found, no conclusion can
be drawn.
- Secondly, there are the ’yes/maybe’ (block B) methods, such as the affine bounds
implemented in the [Decomon library](https://github.com/airbus/decomon). Here, the outputs of a network are
bounded. If the bounds found are within the stability bounds, then the property
is verified. If the bounds are too loose and exceed the required stability interval,
no conclusion can be drawn. So far, these methods called incomplete methods are
partial in the sense that they do not provide an absolute solution to the question of
a network’s stability but are generally fast.
- The last family of methods corresponds to the ’yes/no’ complete methods ((block C)). Examples include [SMT](https://github.com/NeuralNetworkVerification/Marabou) or [MILP](https://gurobi-machinelearning.readthedocs.io/en/stable/index.html) solvers, which provide an exact answer
at the cost of significant computation time.

<div align="center">
    <img src="https://raw.githubusercontent.com/airbus/Airobas/main/docs/source/pipeline.jpg" width="55%" alt="decomon" align="center" />
</div>


## Installation

Quick version:
```shell
pip install .
```
For more details, see the [online documentation](https://airbus.github.io/airobas/main/install).




## Documentation

The latest documentation is available [online](https://airbus.github.io/airobas).

## Examples

Some educational notebooks will be available *soon*.
