# 2048-gym

Work in progress!

## Lessons Learned

### Invalid actions

* A lot of thoughts went into handling invalid actions. Every action that does
not change the board is considered invalid. A naive implementation of a greedy
agent would get stuck if an invalid action has the highest Q value. The general
takeaway ist that it is challenging for the agent to learn what actions are
valid for a given state. Here is what I tried:
** Reward shaping is not the road to success! Giving negative rewards does not
result in any meaningful policy.
** Finishing the episode after an invalid action often leads to policies
avoiding invalid moves, but as the game usually stops early the agent can not
explore it. Resulting policies seem stupid but are a valid local minimum. For
example the A2C agent started alternately executing LEFT and RIGHT.
Additionally the scores were really low as a eps-greedy agents random
explorations stop the game at some point with invalid actions. The agent could
not learn to handle high values.
** Most promising is the implementation of a new policy that takes the validity
of the actions into account. It does not simply select the action with the
highest Q-value but selects the one that is actually valid. This means the
policy exploits the known prior of invalid actions and simply ignores the
invalid ones.

### Model architecture

### A2C Agent

* Training time: It seems like the agent starts to overfit / exploit a
suboptimal policy after some training time. Early stopping can prevent this and
lead to a reasonable policy. For example the highest number is stored in a
corner and values a preferably summed up at the two adjoining borders.

## TODO
[ ] Create Jupyter notebook for better documentation of the test runs
[ ] huskarl.Simulation callback for end of episode, final observation 
-> log high score, highest tile (create nice dotplot with distribution)
[ ] huskarl.Simulation callback to log observation -> action pairs
[ ] huskarl.Simulation n_episodes instead of steps?
[ ] Animate GIF to show how the agent plays
