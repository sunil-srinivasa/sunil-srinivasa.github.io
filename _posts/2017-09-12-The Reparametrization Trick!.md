---
layout: post
title: The Reparameterization Trick!
author: "Sunil Srinivasa"
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<style>
.center-image
{
    margin: 0 auto;
    display: block;
}
</style>
## ARTICLE IN PROGRESS
## Backpropagating through continuous and discrete samples

The proliferation of deep learning and deep reinforcement learning technologies has been made possible largely due to the age-old *backpropagation algorithm*, that was popularized in this [1986 paper](https://www.nature.com/nature/journal/v323/n6088/pdf/323533a0.pdf). Backpropagation (often shortened to backprop) is a technique to calculate the gradients of a loss function in a artificial neural network (typically represented via a parameterized computation graph) with respect to its parameters. These gradients can then be used within the family of stochastic gradient descent (SGD) algorithms (e.g. Vanilla SGD, ADAM, RMSProp) to learn the optimal network parameters. The tandem of backprop and SGD works far simpler and faster than earlier approaches to learning, making it possible to use neural nets to solve problems which were previously deemed insoluble. Researchers have now extensively developed neural network architectures (e.g., for [image classification](https://arxiv.org/abs/1512.03385), [generating captions from images](http://ieeexplore.ieee.org/document/7534740/), [speech recognition](https://arxiv.org/abs/1512.02595) and [policy optimization in reinforcement learning](https://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf)) that are complex owing to the task in hand; yet easily trainable.

While the backprop algorithm sits at the core of present-day machine learning algorithms, it is only sufficient when the loss function is deterministic and a composition of differentiable functions of the network parameters. More often than not, the activation functions used in neural networks (Sigmoid, ReLU, Tanh) are differentiable and deterministic, and so are the typical loss functions (squared loss, cross-entropy loss); thus, the gradients can be computed quite effortlessly. A rich class of optimization problems arising throughout machine learning, however, still deal with loss functions that comprise an expectation over stochastic variables.Two classical examples that observe this formulation include (1) likelihood maximization in the AutoEncoding Variational Bayes Algorithm, and (2) reward maximization in Policy Gradient Algorithms. This blog article provides an exposition on how to tackle such problems.

Speaking mathematically, we focus on the following problem. Consider a neural network with parameters $$\theta$$, and a random variable $$X$$ parameterized by $$\theta$$, i.e., with a probability distribution function $$X \sim p(x;\theta)$$. Now, suppose we want to optimize a loss function $$L=\mathbb{E}_XL(X)$$, averaged over several samples of $$X$$. In order to update $$\theta$$ via SGD, we need to compute the gradient of the loss function with respect to $$\theta$$, i.e.,
$$
\begin{aligned}
\nabla_{\theta}L = \frac{\partial}{\partial\theta}\mathbb{E}_{X \sim p_X(x;\theta)}l(X).
\end{aligned}
$$

If $$p_X(x;\theta)$$ is a continuous function of $$\theta$$, we can write
$$
\begin{aligned}
\frac{\partial}{\partial\theta}\mathbb{E}_{X \sim p_X(x;\theta)}l(X)&=\frac{\partial}{\partial\theta}\int{}l(x)p_X(x;\theta)\text{d}x\\
&=\int{}l(x)\frac{\partial}{\partial\theta}p_X(x;\theta)\text{d}x\\
&=\int{}l(x)\left[\frac{\partial}{\partial\theta}p_X(x;\theta)\frac{1}{p_X(x;\theta)}\right]p_X(x;\theta)\text{d}x\\
&=\int{}l(x)\left[\frac{\partial}{\partial\theta}\log p_X(x;\theta)\right]p_X(x;\theta)\text{d}x\\
&=\mathbb{E}_X\left[l(X)\nabla_{\theta}\log p_X(x;\theta)\right]
\end{aligned}
$$

When
$$
\begin{aligned}
\frac{\partial}{\partial\theta}\mathbb{E}_{X\sim p_Z(z;\theta)}l(x) &=& \frac{\partial}{\partial\theta}\int_{}l(x)p_Z(x;\theta)\text{d}z.
\end{aligned}
$$

Speaking of (2), a Gaussian policy is natural for representing continuous action spaces. For that case, the output of the Gaussian policy network is a linear function of the neural network parameters $$\theta$$, and gives the mean of the agent's policy $$\mu(\theta)$$. With the policy's standard deviation fixed to $$\sigma^2$$, the policy follows $$\pi(\theta)\sim\mathcal{N}(\mu(\theta),\sigma^2)$$. The objective function is  
