# Description
This project implements Bayesian Quadrature on top of GPytorch.

# Intro
Bayesian Quadrature (BQ) is a probabilistic method to approximate an integral in the form  of 

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\int f(x) dx"/>
</p>

or 
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\int f(x)p(x) dx" />.
</p>
The latter integral is usually appeared in Bayesian inference.

BQ can be useful when <img src="https://latex.codecogs.com/gif.latex?f(x)"/> is expensive to compute, prohibited to 
perform Monte Carlo estimation. 

The main idea is to use Gaussian Process (GP) as a surrogate function 
for the true <img src="https://latex.codecogs.com/gif.latex?f(x)"/> and the 
linearity of GP. Since integral is just a linear operator, we can obtain a new GP after apply it over a GP. Let's say

<p align="center">
<img src="https://latex.codecogs.com/gif.latex? f(x) \sim \mathcal{GP}(0, k(x,x'))"/>
</p>

After observing <img src="https://latex.codecogs.com/gif.latex? (x_1, f(x_1)),\dots,(x_n, f(x_n))" />, the integral follows a multivariate normal distribution:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex? \int f(x)p(x) \sim \mathcal{N}(\mu, \sigma^2)"/>
</p>
where
<p align="center">
<img src="https://latex.codecogs.com/gif.latex? \mu = \mathbf{z}^\top \mathbf{K}^{-1} \mathbf{f}"/>
</p>
and 

<p align="center">
<img src="https://latex.codecogs.com/gif.latex? \sigma^2 = \int \int k(x,x')p(x)p(x')dx dx' - \mathbf{z}^\top \mathbf{K}^{-1} \mathbf{z}"/>
</p>
with <img src="https://latex.codecogs.com/gif.latex? \mathbf{z} = [z_{1:n}]^\top, z_i = \int k(x, x_i)p(x)dx"/>.

Discussion on limitations of BQ:
-  <img src="https://latex.codecogs.com/gif.latex? p(x)"/> is restricted to a certain family (normal, uniform)) to have closed-formed solutions for  <img src="https://latex.codecogs.com/gif.latex? z_i"/>
- BQ is applied in low-dimensional spaces rather than high-dimensional settings.

# Requirements

```angular2
Python >= 3.6
Pytorch ==1.3
GPytorch == 0.3.6
```