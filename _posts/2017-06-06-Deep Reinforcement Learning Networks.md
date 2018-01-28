---
layout: post
title: Deep Reinforcement Learning Networks
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

***Deep Reinforcement Learning Papers and Summaries***

## 1.  [Deep Q-Network (DQN) - NIPS Deep Learning Workshop, 2013](https://arxiv.org/abs/1312.5602 "Playing Atari with Deep Reinforcement Learning")

  **Summary:**
  - Combined deep network with Q-learning for solving discrete action Atari tasks.
  - Neural network with weights $$\theta$$ for to estimate $$Q(s,a)$$.

  **Tricks:**
  - **Experience Replay** from a **replay memory buffer** is used to alleviate the problems of correlated data and non-stationary distributions. This results in greater data efficiency, breaks correlations between samples and smoothens learning by avoiding oscillations or divergences in the parameters.

  **Equations:**
  - $$\begin{aligned}
      Q^*(s,a)= r+\gamma\mathbb{E}_{s'}[\max_{a'}Q^*(s',a')\big | s,a].
    \end{aligned}$$
  - Use a non-linear function approximator (deep neural network) to estimate the action-value function $$Q(s,a;\theta)\approx Q^*(s,a)$$.
  - The loss function at iteration $$i$$ is $$L_i(\theta_i)=\mathbb{E}_{s,a}\left[(y_i-Q(s,a;\theta_i))^2\right]$$, where $$\begin{aligned}
   y_i=r+\gamma\mathbb{E}_{s'}[\max_{a'}Q(s',a';\theta_{i-1})\big|s,a]
   \end{aligned}$$
   is the target for iteration $$i$$.
  - $$\nabla_{\theta_i}L_i(\theta_i) = \mathbb{E}_{s,a}\left[\left(r+\gamma\max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i)\right)\nabla_{\theta_i}Q(s,a;\theta_i)\right]$$.

  **Experiments:**
  - $$7$$ games in the Atari $$2600$$ suite.
  - Superior in comparison with Random, Human, SARSA, Contingency, HNeat best and HNeat Pixel.

  **Other Comments:**
  - **Model-free** and **Off-policy**.
  - Accelerated running time by repeating actions taken $$4$$ times. Via this technique, roughly $$4$$ more experiments can be run in the same time.
  - Input taken to be $$4$$ consecutive frames of $$84\times 84$$.
  - **Architecture**: Two convolutional layers: $$16$$ $$8\times 8$$ filter stride $$4$$ followed by a $$32$$ $$4\times 4$$ filters with stride $$2$$ followed by fully connected layer with $$256$$ nodes.
  - One network that inputs $$s$$ and outputs $$Q(s,a)$$ for each action, rather than running through a network that takes inputs $$(s,a)$$ multiple times.
<br><br>

## 2. [DQN - Nature 2015](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html "Human-level control through deep reinformcement learning")

  **Tricks:**
  - **Separate target network** for generating $$y_i$$ (which is cloned with the $$Q$$-network every $$C$$ updates).
  - **Gradient clipping**.
  - Like in the NIPS paper, **experience replay**.

  **Experiments:**
  - $$49$$ games in the Atari $$2600$$ suite.
  - Superior in comparison with Random, Human and SARSA.

  **Other Comments:**
  - **Architecture**: Three convolutional layers: $$32$$ $$8\times 8$$ filter stride $$4$$ followed by a $$64$$ $$4\times 4$$ filters with stride $$2$$ followed by $$64$$ $$3\times 3$$ filter with stride $$1$$ followed by fully connected layer with $$512$$ nodes.
<br><br>

## 3. [Deep Recurrent Q-Network (DRQN) - 2015](https://arxiv.org/abs/1507.06527 "Deep Recurrent Q-Learning for Partially Observable MDPs")

  **Summary:**
  - Combined DQN architecture with **LSTM** on top.

  **Tricks:**
  - Outputs from the third convolutional layer are fed into a fully-connected LSTM layer with $$512$$ cells. The input to the DRQN is a single frame, but inputs need to be sequential frames (because of using the LSTM), and not randomly selected.

  **Experiments:**
  - $$10$$ games in the Atari $$2600$$ suite with a focus on Pong, with and without flickering.
  - Better performance in comparison with DQN for some games.
<br><br>

## 4. [Prioritized Experience Replay - ICLR 2016](https://arxiv.org/abs/1511.05952 "Prioritized Experience Replay")

  **Main Idea:**
  - Rather than replaying samples uniformly, prioritize them based on the magnitude of the TD error associated with them.

  **Tricks:**
  - **Stochastic proioritization** to alleviate loss of diversity
  - **Weighted Importance sampling** to correct the introduced bias.

  **Experiments:**
  - Outperformed DQN in $$41$$ out of $$49$$ Atari games.
<br><br>

## 5. [Asynchronous Advantage Actor-Critic (A3C) - ICML 2016](https://arxiv.org/abs/1602.01783 "Asynchronous Methods for Deep Reinforcement learning")

  **Summary:**
  - Asynchronous variants of four standard RL algorithms (one-step SARSA, one-step Q-learning, $n$-step Q-learning, advantage actor-critic).
  - Show that parallel actor-learners have a stabilizing effect on training.

  **Tricks:**
  - No experience replay; instead, asynchronously executes multiple agents in parallel. Experiments can run on a multi-core CPU, no need for GPUs.
  - Entropy of the policy (or differential entropy for continuous tasks) is added to the loss function in order to improve exploration by discouraging premature convergence to suboptimal deterministic policies.
  - For the discrete action tasks, the action output is softmax, whereas for the continuous task, the output is the mean and variance of the normal distribution. There is another network (with a different set of parameters) that outputs the value function $V$.

  **Equations:**
  - For Q-learning methods, we use  the gradients $$\partial \left(y_i-Q(s_i,a_i;\theta')\right)^2/\partial \theta'$$.
  - For the actor-critic method, we use the gradients $$\begin{aligned}
  \nabla_{\theta'}\log \pi(a_i|s_i;\theta')(y_i-V(s_i;\theta'_v))
  \end{aligned}$$
  for the actor and $$\begin{aligned}
  \partial(y_i-V(s_i,\theta'_v))^2/\partial \theta'_v
  \end{aligned}$$ for the critic.

  **Experiments:**
  - $$57$$ Atari games, TORCS Car Racing Simulator, Mujoco and Labyrinth
  - Comparison made w.r.t. DQN and Human tester.

  **Other Comments:**
  - **Architecture:**<br>
  Atari: The agents used the network architecture from (Mnih et al., 2013) as well as a recurrent agent with an additional $$256$$  LSTM  cells after the final hidden layer.<br>
  Mujoco: In the low dimensional physical state case, the inputs are mapped to a hidden state using $$1$$ hidden layer with $$200$$  ReLU  units. In the pixels case, the input was passed through $$2$$  conv  layers without any non-linearity or pooling. In either case, the output of the encoder layers were fed to a single layer of $$128$$ LSTM cells.<br>
  Labyrinth: A3C LSTM agent trained on this task using only $$84\times84$$ RGB images as input.
<br><br>

## 6. [Dueling network architectures - ICML 2016](https://arxiv.org/abs/1511.06581 "Dueling Network Architectures for Deep Reinforcement Learning")

  **Summary:**
  - Two streams to separately estimate the scalar value function $$V(s;\theta,\alpha)$$ and the vector advantage function (for each action), $$A(s,a;\theta,\beta)$$. The streams (networks) share the parameters ($$\theta$$), and also have separate parameters ($$\alpha$$ and $$\beta$$). The sum of both these networks provides the Q-function values $$Q(s,a;\theta,\alpha,\beta)$$.
  - With every update of the $$Q$$ values in the dueling architecture, the value stream $$V$$ is updated - this contrasts with the updates in a single-stream architecture where only the value for one of the actions is updated, the value for all other actions remain untouched. This more frequent updating leads to a faster convergence.

  **Tricks:**
  - $$Q$$ cannot recover $$V$$ and $$A$$ uniquely, i.e., there exits the *unidentifiablity* problem. To address this issue, the authors add a constant to the advantage function (see equation section below). This loses the original semantics of $$V$$ and $$A$$ but increases the stability of the operation.

  **Equations:**
  -$$\begin{aligned} Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\beta)+\left(A(s,a;\theta,\alpha)-\frac{1}{|\mathcal{A}|\sum_{a'}A(s,a';\theta,\alpha)}\right)
  \end{aligned}$$.

  **Experiments:**
  - $$57$$ games in the Atari $$2600$$ suite.
<br><br>

## 7. [Normalized Advantage Functions (NAF) - ICML 2016](https://arxiv.org/abs/1603.00748 "Continuous Deep Q-Learning with Model-based Acceleration")

  **Summary:**
  - Continuous variant of the $$Q$$-learning algorithm using normalized advantage functions.
  - Use learned models for accelerating model-free RL algorithms.

  **Tricks:**
  - Replay buffer
  - Imagination rollouts

  **Equations:**
  - $$Q(s,a;\theta^Q)=A(s,a;\theta^A)+V(s;\theta^V)$$ .
  - $$A(s,a;\theta^A)=-\frac{1}{2}\left(a-\mu(s;\theta^{\mu})\right)^TP(s;\theta^P)\left(a-\mu(s;\theta^{\mu})\right)$$, where $$P$$ is a state-dependent positive-definite square matrix.
  - $$P(s;\theta^P)=L(s;\theta^P)L(s;\theta^P)^T$$, where $$L$$ is a lower-triangular matrix whose entries come from a linear output layer of a neural network, with the diagonal terms exponentiated.
  - Since the $$Q$$-function is quadratic in $$a$$, the action that maximizes the $$Q$$-function is always $$\mu(s;\theta^{\mu})$$.

## 8. [Policy Gradient and Q-Learning (PGQ) - ICLR 2017](https://arxiv.org/abs/1611.01626 "PGQ: Combining Policy Gradient and Q-Learning")

  **Summary:**
  - Combined policy gradient and $$Q$$-learning
  -

  **Tricks:**
  - Relate $$\begin{aligned}
  \pi(a | s)
  \end{aligned}$$ and $$Q(s,a)$$.

  **Equations:**
  - $$ \Delta\theta \propto \mathbb{E}_{s,a}Q^{\pi}(s,a)\nabla_{\theta}\log\pi(s,a) + \alpha \mathbb{E}_s\nabla_{\theta}H(\pi_s)$$, after adding entropy regularizer.
  - But $$\begin{aligned}
  \nabla_{\theta}H(\pi_s) = -\sum_a\pi_{\theta}(a|s)\left(\log\pi_{\theta}(a|s)+1\right)\nabla_{\theta}\log\pi_{\theta}(a|s)
  \end{aligned}$$.
  - Therefore, $$\begin{aligned}
  \Delta\theta \propto \mathbb{E}_{s,a}\left(Q^{\pi}(s,a)-\alpha\log\pi(a|s)-\alpha\right)\nabla_{\theta}\log\pi(s,a)
  \end{aligned}$$
  - In the tabular case, $$\pi(a|s) = \exp\left(A^{\pi}(s,a)/\alpha-H(\pi_s)\right)$$, i.e., the policy is a softmax over the advantage function induced by that policy, where the regularization parameter $$\alpha$$ can be interpreted as temperature.
  -

  **Experiments:**

  **Other Comments:**
<br><br>

## 9. [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971 "Continuous Control with Deep Reinforcement Learning")

## 10. [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477 "Trust Region Policy Optimization")

## 11. [Policy Gradient Q-Learning (PGQL)](https://arxiv.org/abs/1611.01626 "Combining policy gradient and Q-learning")

## 12. [Q-PROP](https://arxiv.org/abs/1611.02247 "Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic")

## 13. [Actor-Critic with Experience Replay (ACER)](https://arxiv.org/abs/1611.01224 "Sample Efficient Actor-Critic with Experience Replay")

## 14. [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.0634 "Proximal Policy Optimization Algorithms")

## 15. [Evolutionary Strategies (ES)](https://arxiv.org/abs/1703.03864"Evolution Strategies as a Scalable Alternative to Reinforcement Learning")

## 16. [Policy Gradients = Q-Learning](https://arxiv.org/abs/1704.06440 "Equivalence Between Policy Gradients and Soft Q-Learning")

## 17. [Path Consistency Learning (PCL)](https://arxiv.org/abs/1702.08892 "Bridging the Gap Between Value and Policy Based Reinforcement Learning")
