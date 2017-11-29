
---
layout: post
title: Multi-objective Reinforcement Learning
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

## How can we handle a vector of rewards in the context of reinforcement learning?

**What is Multi-objective Reinforcement Learning**

Reinforcement learning is classically known to optimize a policy that maximizes a (scalar) reward function. However, in many problems, we encounter several objectives or rewards that we care about; sometimes, the objectives are conflicting with one another. For example, in robotic locomotion, we want to maximize forward velocity but also minimize joint torque and impact with the ground. The subfield of reinforcement learning that deals with multiple objectives, i.e., a vector reward function rather than a scalar, is called Multi-objective reinforcement learning (MORL)

In the MORL domain, there are two standard approaches that are usually taken:

1) The $\textbf{single-objective}$ practice is to use a scalar objective function that is a weighted sum or a function of all the objectives. In this regard, it is sometimes common to order or rank the objectives for choosing the appropriate weights; and also order or rank the solutions obtained. However, it is not only difficult to determine how to weigh the objectives, but also often harder to balance the factors to achieve satisfactory performance along all the objectives.

2) The alternative $\textbf{Pareto}$ starategy tries to find multiple solutions to the MORL problem that offer trade-offs among the various objectives. In other words, these multiple solutions, also called $\textbf{Pareto solutions}$, are non-superior or non-dominating over each other. The set of Pareto-optimal solutions constitute what is called the $\textbf{Pareto front}$ or $\textbf{Pareto boundary}$. It is left to the discretion of the end-user to then select the operating solution point. Pareto methods are also called filter methods (see [[1]](http://users.iems.northwestern.edu/~nocedal/book/index.html){:target="_blank"}, chapter 15.4), which are classical algorithms from multi-objective optimization literature that seek to generate a sequence of points, so that each one is not strictly dominated by a previous one .

In this blog post, we explain how to obtain the Pareto front for the [Cartpole](https://gym.openai.com/envs/CartPole-v0/){:target="_blank"} environment.

**The Cartpole environment**

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track (see the gif below). The system is controlled by applying a force between $$+10$$ to $$-10$$ to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of $$+10$$ is provided for every timestep that the pole remains upright. The episode ends when the pole is more than $$15$$ degrees from vertical, or the cart moves more than $$2.4$$ units from the center.

![The Cartpole environment.]({{site.baseurl}}/assets/images/2016-12-10-MORL/cartpole.gif){: .center-image}

Current implementations of the Cartpole environment on well-known frameworks such as [rllab](https://github.com/rll/rllab){:target="_blank"} compute consider three separate terms (or objectives) for the reward function: a constant reward, ucost and xcost.

1) $$\mathbf{10}$$: this is a constant reward that is provided for every instant that the cart is upright.

2) $$\mathbf{ucost$$ = $$-1e-5*(action^2)}$$ : this is a cost that takes the action into account. Higher the action, the more negative this objective. Ideally, we want to apply as little force on the cart as required to make it stand.

3) $$\mathbf{xcost = -(1 - np.cos(pole_angle))}$$ : this is a cost that is based on how upright the cart is, the more vertical the better. Note that the angle here is measured from the pole's upright position. Accordingly, when $$pole_angle=0$$, $$xcost = 0$$, while $$xcost=-1$$ for a fallen pole that's lying on the ground.

Along the lines of the single-objective or scalar reward function, the overall reward is simply taken to be the sum of the three terms, i.e., $$10+ucost+xcost$$, and is a scalar.

***A methodology for considering the rewards separately***

In the context of multi-objective optimization, there does not typically exist a feasible solution that minimizes (or maximizes) all objective functions simultaneously. Therefore, attention is paid to *Pareto-optimal* (or *Pareto-dominant*) solutions, which are defined to be solutions that cannot be improved in any of the objectives without degrading at least one of the other objectives.

Let us focus on a two-objective optimization problem for the purpose of understanding where we are looking to maximize both the objectives. Consider two distinct solutions on this two-dimensional space: $$S_1 = (x_1,y_1)$$ and $$S_2 = (x_2,y_2)$$. $$S_1$$ is said to (pareto)-dominate solution $$S_2$$ if
* $$x_1 > x_2$$ and $$y_1 \geq y_2$$

OR

* $$x_1 \geq x_2$$ and $$y_1 > y_2$$

The figure below provides a depiction of a Pareto front for the two-dimensional case. Notice that there are multiple solutions to this two-objective problem, that are marked blue. However, while most of these solutions are dominated by other solutions; it is only the red points that are non-dominated, and those points form the Pareto boundary (or front). Points under the Pareto front are feasible while those beyond the Pareto front are infeasible.


![alt text](http://pubs.rsc.org/services/images/RSCpubs.ePlatform.Service.FreeContent.ImageService.svc/ImageService/Articleimage/2010/CP/b914552d/b914552d-f4.gif "Pareto front depiction")

In this following, we delve on the Pareto strategy. Specifically, we split this into two separate reward functions or objectives, xcost and ucost (Note: 10 is just a constant that can be added on top). We aim to find an approximation of the Pareto frontier by finding points that are not strictly dominated by any other.

### Algorithm

We use the $\textbf{radial algorithm}$ approach presented in Parisi et al., ``Policy gradient approaches for multi-objective sequential decision making,'' IEEE International Joint Conference on Neural Networks (IJCNN), July 2014.

The idea behind this algorithm is the following:
Consider the two extreme steepest ascent directions (one for each objective) that maximize each objective and neglect the other objective. These directions are given by $\theta_1=\nabla{_\theta} J_1(\theta)$ and $\theta_2=\nabla{_\theta} J_2(\theta)$, where $J_i=\mathbb{E}R_i$, and $R_i$ is the reward along axis $i$. Any direction in between $\theta_1$ and $\theta_2$ will simultaneously increase both the objectives. As a consequence, a sampling of directions amidst the two extreme directions corresponds to pointing at different locations on the Pareto frontier. Every direction intrinsically defines a preference ratio over the two objectives. We uniformly sample the ascent direction space via a parameter $\lambda\in\{0,1\}$ and use the ascent direction
$\theta_{\lambda}=\lambda\times\theta_1+(1-\lambda)\times\theta_2$.

Equivalently, we may set the overall reward to $R=\lambda\times R_1 + (1-\lambda)\times R_2$ and perform policy optimization with this modified reward function. This is because

$\theta_{\lambda}=\lambda\times\nabla{_\theta} J_1(\theta)+(1-\lambda)\times\nabla{_\theta} J_2(\theta)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=\nabla_{\theta}(\lambda\times J_1(\theta)+(1-\lambda)\times J_2(\theta))$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=\nabla_{\theta}(\lambda\times\mathbb{E}R_1+(1-\lambda)\times\mathbb{E}R_2$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=\nabla_{\theta}\mathbb{E}R$,

where $R=\lambda\times R_1 + (1-\lambda)\times R_2$

Each direction provides an unique solution to our optimization problem. By determining the set of non-dominated solutions, the Pareto boundary can be well approximated.

The psuedo-code for the algorithm for a two-dimensional reward function is as follows:


* $\{\lambda_i\}_{i=1}^p$ <-- uniform sampling of $[0,1]$.

* $\textbf{for}$ $i=1,\ldots,p$ $\textbf{do}$:

 * Initialize policy network parameter $\theta$ randomly.
    
 * $\textbf{for}$ iteration in $1,\ldots,N$ $\textbf{do}$:
    
   * Collect trajectories with features (State, Action, Reward:(ucost,xcost), nextState).
        
   * Set net reward $R=\lambda\times R_1+(1-\lambda)\times R_2$.
    
   * Implement a policy optimization algorithm using reward function $R$.
        
 * Record the optimal value $\theta_i$ for $\lambda_i$.
    
* Determine the set of Pareto-optimal points, i.e., the set of objective values that are not dominated by one another.
* The Patero front is obtained by piecewise-linearly connecting the set of Pareto-optimal points obtained. Each point on any of these lines is attainable by time sharing between the end points of that line.

***References***

[1] Jorge Nocedal and Stephen J. Wright, [Numerical Optimization](http://users.iems.northwestern.edu/~nocedal/book/index.html)


```python

```

### Implementation of the Cartpole problem with VPG


```python
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def MORL(lambda_array):
    from cartpole_env import CartpoleEnv

    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from normalized_env import normalize
    import numpy as np
    import theano
    import theano.tensor as TT
    from lasagne.updates import adam

    # normalize() makes sure that the actions for the environment lies
    # within the range [-1, 1] (only works for environments with continuous actions)
    env = normalize(CartpoleEnv())

    # We will collect N trajectories per iteration
    N = 200
    # Each trajectory will have at most T time steps
    T = 100
    # Number of iterations
    n_itr = 50
    # Set the discount factor for the problem
    discount = 0.99
    # Learning rate for the gradient update
    learning_rate = 0.01

    # Construct the computation graph
    # Initialize a neural network policy with a single hidden layer of 8 hidden units
    policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,32))
    # Initialize a linear baseline estimator using default hand-crafted features
    baseline = LinearFeatureBaseline(env.spec)

    policy_init = policy.get_param_values()
    baseline_init = baseline.get_param_values()

    # Create a Theano variable for storing the observations
    # We could have simply written `observations_var = TT.matrix('observations')` instead for this example.
    # However, doing it in a slightly more abstract way allows us to delegate to the environment for handling
    # the correct data type for the variable. For instance, for an environment with discrete observations,
    # we might want to use integer types if the observations are represented as one-hot vectors.
    observations_var = env.observation_space.new_tensor_variable(
        'observations',
        # It should have 1 extra dimension since we want to represent a list of observations
        extra_dims=1
    )
    actions_var = env.action_space.new_tensor_variable(
        'actions',
        extra_dims=1
    )
    advantages_var = TT.vector('advantages')

    # policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
    # distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
    dist_info_vars = policy.dist_info_sym(observations_var)

    # policy.distribution returns a distribution object under rllab.distributions. It contains many utilities
    # for computing distribution-related quantities, given the computed dist_info_vars. Below we use
    # dist.log_likelihood_sym to compute the symbolic log-likelihood. For this example, the corresponding
    # distribution is an instance of the class rllab.distributions.DiagonalGaussian
    dist = policy.distribution

    # Note that we negate the objective, since most optimizers assume a
    # minimization problem
    surr = - TT.mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * advantages_var)

    # Get the list of trainable parameters.
    params = policy.get_params(trainable=True)
    grads = theano.grad(surr, params)

    f_train = theano.function(
        inputs=[observations_var, actions_var, advantages_var],
        outputs=None,
        updates=adam(grads, params, learning_rate=learning_rate),
        allow_input_downcast=True
    )

    solutions = []
    solutions_iter = []
    for l in lambda_array:
        policy.set_param_values(policy_init)
        baseline.set_param_values(baseline_init)

        for itr in xrange(n_itr):
            # Initialize paths list
            paths = []

            for _ in xrange(N):
                observations = []
                actions = []
                rewards = []
                rewardsC = [] # constant reward
                rewardsX = [] # xcost
                rewardsU = [] # ucost

                observation = env.reset()

                for _ in xrange(T):

                    action, _ = policy.get_action(observation)
                    next_observation, Reward, terminal, _ = env.step(action)
                    reward, rewardC, rewardX, rewardU = Reward

                    reward = l*(rewardC + rewardX) + (1-l)*rewardU # weighted reward

                    observations.append(observation)
                    actions.append(action)
                    rewards.append(reward)

                    rewardsC.append(rewardC)
                    rewardsX.append(rewardX)
                    rewardsU.append(rewardU)

                    observation = next_observation
                    if terminal:
                        # Finish rollout if terminal state reached
                        break

                # We need to compute the empirical return for each time step along the
                # trajectory
                path = dict(
                    observations=np.array(observations),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    rewardsC=np.array(rewardsC),
                    rewardsX=np.array(rewardsX),
                    rewardsU=np.array(rewardsU)
                )
                path_baseline = baseline.predict(path)
                advantages = []
                returns = []
                return_so_far = 0
                for t in xrange(len(rewards) - 1, -1, -1):
                    return_so_far = rewards[t] + discount * return_so_far
                    returns.append(return_so_far)
                    advantage = return_so_far - path_baseline[t]
                    advantages.append(advantage)
                # The advantages are stored backwards in time, so we need to revert it
                advantages = np.array(advantages[::-1])
                # And we need to do the same thing for the list of returns
                returns = np.array(returns[::-1])

                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

                path["advantages"] = advantages
                path["returns"] = returns

                paths.append(path)

            baseline.fit(paths)

            observations = np.concatenate([p["observations"] for p in paths])
            actions = np.concatenate([p["actions"] for p in paths])
            advantages = np.concatenate([p["advantages"] for p in paths])
            
            AvgReturn = np.mean([sum(p["rewards"]) for p in paths])
            AvgReturnC = np.mean([sum(p["rewardsC"]) for p in paths])
            AvgReturnX = np.mean([sum(p["rewardsX"]) for p in paths])
            AvgReturnU = np.mean([sum(p["rewardsU"]) for p in paths])
            
            solutions_iter.append(AvgReturn)

            f_train(observations, actions, advantages)
        #print('Iteration: {}; AverageReturn: {}; AverageReturnU: {}, AverageReturnX: {}'
        #.format(itr,AvgReturn,AvgReturnU,AvgReturnX))

        solutions.append([AvgReturnC,AvgReturnX,AvgReturnU,AvgReturn])
    return solutions, solutions_iter
```

### Results when using a scalar reward function


```python
lambda_array = [0.5]
solutions, solutions_iter = MORL(lambda_array)
```

    /home/sunil/anaconda2/envs/rllab/lib/python2.7/site-packages/theano/tensor/signal/downsample.py:6: UserWarning: downsample module has been moved to the theano.tensor.signal.pool module.
      "downsample module has been moved to the theano.tensor.signal.pool module.")



```python
plt.plot([x*2 for x in solutions_iter],'o-',linewidth=2)
plt.xlabel('Iteration #')
plt.ylabel('Net reward')
plt.grid()
```


![png](output_14_0.png)


### Finding the Pareto optimal points considering two objectives

The total reward can be decomposed as $R_1=10+xCost$ and $R_2=uCost$. We use these as our two objectives in this section. We choose only two objectives here since the Pareto boundary can be easily depicted and explained using a two-dimensional plot.


```python
lambda_array = np.arange(0.0,1.01,0.01)
solutions, solutions_iter = MORL(lambda_array)
```


```python
# Returns
retC = [x[0] for x in solutions]
retX = [x[1] for x in solutions]
retU = [x[2] for x in solutions]

# Normalizing returns in [0,1]
ret1 = list(np.array(retC) + np.array(retX))
ret2 = retU

ret1 = (ret1-min(ret1))/(max(ret1)-min(ret1))
ret2 = (ret2-min(ret2))/(max(ret2)-min(ret2))
```


```python
# Pareto frontier computation for 2D
def pareto_frontier(Xs, Ys):
    Xs, Ys = zip(*sorted(zip(Xs, Ys),reverse=True))
    p_front = []
    p_front.append([Xs[0],Ys[0]])
    for idx in xrange(1,len(Xs)):
        for j in xrange(0,idx):
            if Ys[idx] < Ys[j]:
                break;
        else:
            p_front.append([Xs[idx],Ys[idx]])

    return [p[0] for p in p_front], [p[1] for p in p_front]
```


```python
p1, p2 = pareto_frontier(ret1, ret2)
```


```python
import matplotlib
from matplotlib import gridspec
import matplotlib.patches as patches

matplotlib.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(20,8)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

color1 = 'lightblue'
color2 = 'yellow'

ax = plt.subplot(gs[0])
plt.plot(ret1,ret2,'ro',markersize=12)
plt.grid()
plt.xlabel('Normalized (10 + xCost)')
plt.ylabel('Normalized uCost')
plt.xlim(0,1)
plt.ylim(0,1)

rect = matplotlib.patches.Rectangle((0.8,0.0), 0.2, 0.4, fill = None, edgecolor=color2,linewidth=15)
ax.add_patch(rect)

for i in xrange(len(p1)-1):
    plt.plot([p1[i],p1[i+1]],[p2[i],p2[i+1]],'k-',linewidth=3)
plt.fill_between(p1, p2, 0, color=color1)
plt.title('Pareto front and the achievable region')

plt.legend()
plt.text(0.25,0.25, 'Achievable Region',fontsize=24)

plt.subplot(gs[1])
plt.plot(ret1,ret2,'ro',markersize=12)
plt.grid()
plt.xlabel('Normalized (10 + xCost)')
plt.ylabel('Normalized uCost')
plt.xlim(0.8,1.0)
plt.ylim(0.0,0.4)

for i in xrange(len(p1)-1):
    plt.plot([p1[i],p1[i+1]],[p2[i],p2[i+1]],'k-',linewidth=3)

plt.fill_between(p1, p2, 0, color=color2)

plt.title("Zoomed-in plot")
plt.tight_layout()
```


![png](output_20_0.png)


### Finding the Pareto Optimal points considering the three rewards separately

Here, we decompose the total reward into $R_1=10$, $R_2=xCost$ and $R_3=uCost$. The weighting factors for the rewards $(\lambda_1,\lambda_2)$ now is uniformly sampled over the isosceles right triangle with vertices at $[0,0]$, $[0,1]$ and $[1,0]$. For the policy gradient, we use the reward function $R=\lambda_1\times R_1 + \lambda_2\times R_2+(1-\lambda-1-\lambda_2)\times R_3$.


```python
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def MORL(lambda1_array):
    from cartpole_env import CartpoleEnv

    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from normalized_env import normalize
    import numpy as np
    import theano
    import theano.tensor as TT
    from lasagne.updates import adam

    # normalize() makes sure that the actions for the environment lies
    # within the range [-1, 1] (only works for environments with continuous actions)
    env = normalize(CartpoleEnv())

    # We will collect N trajectories per iteration
    N = 200
    # Each trajectory will have at most T time steps
    T = 100
    # Number of iterations
    n_itr = 50
    # Set the discount factor for the problem
    discount = 0.99
    # Learning rate for the gradient update
    learning_rate = 0.01

    # Construct the computation graph
    # Initialize a neural network policy with a single hidden layer of 8 hidden units
    policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,32))
    # Initialize a linear baseline estimator using default hand-crafted features
    baseline = LinearFeatureBaseline(env.spec)

    policy_init = policy.get_param_values()
    baseline_init = baseline.get_param_values()

    # Create a Theano variable for storing the observations
    # We could have simply written `observations_var = TT.matrix('observations')` instead for this example.
    # However, doing it in a slightly more abstract way allows us to delegate to the environment for handling
    # the correct data type for the variable. For instance, for an environment with discrete observations,
    # we might want to use integer types if the observations are represented as one-hot vectors.
    observations_var = env.observation_space.new_tensor_variable(
        'observations',
        # It should have 1 extra dimension since we want to represent a list of observations
        extra_dims=1
    )
    actions_var = env.action_space.new_tensor_variable(
        'actions',
        extra_dims=1
    )
    advantages_var = TT.vector('advantages')

    # policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
    # distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
    dist_info_vars = policy.dist_info_sym(observations_var)

    # policy.distribution returns a distribution object under rllab.distributions. It contains many utilities
    # for computing distribution-related quantities, given the computed dist_info_vars. Below we use
    # dist.log_likelihood_sym to compute the symbolic log-likelihood. For this example, the corresponding
    # distribution is an instance of the class rllab.distributions.DiagonalGaussian
    dist = policy.distribution

    # Note that we negate the objective, since most optimizers assume a
    # minimization problem
    surr = - TT.mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * advantages_var)

    # Get the list of trainable parameters.
    params = policy.get_params(trainable=True)
    grads = theano.grad(surr, params)

    f_train = theano.function(
        inputs=[observations_var, actions_var, advantages_var],
        outputs=None,
        updates=adam(grads, params, learning_rate=learning_rate),
        allow_input_downcast=True
    )

    solutions = []
    solutions_iter = []
    for l1 in lambda_array:
        for l2 in np.arange(0,1.01-l1,0.1):
            policy.set_param_values(policy_init)
            baseline.set_param_values(baseline_init)

            for itr in xrange(n_itr):
                # Initialize paths list
                paths = []

                for _ in xrange(N):
                    observations = []
                    actions = []
                    rewards = []
                    rewardsC = [] # constant reward
                    rewardsX = [] # xcost
                    rewardsU = [] # ucost

                    observation = env.reset()

                    for _ in xrange(T):

                        action, _ = policy.get_action(observation)
                        next_observation, Reward, terminal, _ = env.step(action)
                        reward, rewardC, rewardX, rewardU = Reward

                        reward = l1*rewardC + l2*rewardX + (1-l1-l2)*rewardU # weighted reward

                        observations.append(observation)
                        actions.append(action)
                        rewards.append(reward)

                        rewardsC.append(rewardC)
                        rewardsX.append(rewardX)
                        rewardsU.append(rewardU)

                        observation = next_observation
                        if terminal:
                            # Finish rollout if terminal state reached
                            break

                    # We need to compute the empirical return for each time step along the
                    # trajectory
                    path = dict(
                        observations=np.array(observations),
                        actions=np.array(actions),
                        rewards=np.array(rewards),
                        rewardsC=np.array(rewardsC),
                        rewardsX=np.array(rewardsX),
                        rewardsU=np.array(rewardsU)
                    )
                    path_baseline = baseline.predict(path)
                    advantages = []
                    returns = []
                    return_so_far = 0
                    for t in xrange(len(rewards) - 1, -1, -1):
                        return_so_far = rewards[t] + discount * return_so_far
                        returns.append(return_so_far)
                        advantage = return_so_far - path_baseline[t]
                        advantages.append(advantage)
                    # The advantages are stored backwards in time, so we need to revert it
                    advantages = np.array(advantages[::-1])
                    # And we need to do the same thing for the list of returns
                    returns = np.array(returns[::-1])

                    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

                    path["advantages"] = advantages
                    path["returns"] = returns

                    paths.append(path)

                baseline.fit(paths)

                observations = np.concatenate([p["observations"] for p in paths])
                actions = np.concatenate([p["actions"] for p in paths])
                advantages = np.concatenate([p["advantages"] for p in paths])

                AvgReturn = np.mean([sum(p["rewards"]) for p in paths])
                AvgReturnC = np.mean([sum(p["rewardsC"]) for p in paths])
                AvgReturnX = np.mean([sum(p["rewardsX"]) for p in paths])
                AvgReturnU = np.mean([sum(p["rewardsU"]) for p in paths])

                solutions_iter.append(AvgReturn)

                f_train(observations, actions, advantages)
            #print('Iteration: {}; AverageReturn: {}; AverageReturnU: {}, AverageReturnX: {}'
            #.format(itr,AvgReturn,AvgReturnU,AvgReturnX))

            solutions.append([AvgReturnC,AvgReturnX,AvgReturnU,AvgReturn])
    return solutions, solutions_iter
```


```python
lambda_array = np.arange(0.0,1.01,0.1)
solutions, solutions_iter = MORL(lambda_array)
```


```python
# Returns
retC = [x[0] for x in solutions]
retX = [x[1] for x in solutions]
retU = [x[2] for x in solutions]

# Normalizing returns in [0,1]
ret1 = retC 
ret2 = retX
ret3 = retU

ret1 = (ret1-min(ret1))/(max(ret1)-min(ret1))
ret2 = (ret2-min(ret2))/(max(ret2)-min(ret2))
ret3 = (ret3-min(ret3))/(max(ret3)-min(ret3))
```


```python
# Pareto frontier computation for 3D
def pareto_frontier(Xs, Ys, Zs=None):
    Xs, Ys, Zs = zip(*sorted(zip(Xs, Ys, Zs),reverse=True))
    p_front = []
    p_front.append([Xs[0],Ys[0], Zs[0]])
    for idx in xrange(1,len(Xs)):
        for j in xrange(0,idx):
            if Ys[idx] < Ys[j] and Zs[idx] < Zs[j]:
                break;
        else:
            p_front.append([Xs[idx],Ys[idx], Zs[idx]])

    return [p[0] for p in p_front], [p[1] for p in p_front], [p[2] for p in p_front]
```


```python
p1, p2, p3 = pareto_frontier(ret1, ret2, ret3)
```


```python
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(p1, p2, p3,'r^')
ax.set_xlabel('\n'+'Normalized constant cost')
ax.set_ylabel('\n'+'Normalized xCost')
ax.set_zlabel('\n'+'Normalized uCost')
```




    <matplotlib.text.Text at 0x7f346bea9bd0>




![png](output_28_1.png)

