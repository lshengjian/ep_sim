# noqa
"""
# Knights Archers Zombies ('KAZ')

This environment is part of the <a href='..'>butterfly environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.butterfly import knights_archers_zombies_v10` |
|----------------------|----------------------------------------------------------------|
| Actions              | Discrete                                                       |
| Parallel API         | Yes                                                            |
| Manual Control       | Yes                                                            |
| Agents               | `agents= ['dispatch', 'crane_1', 'crane_2', 'crane_3']`     |
| Agents               | 4                                                              |
| Action Shape         | (1,)                                                           |
| Action Values        | [0, 5]                                                         |
| Observation Shape    | (512, 512, 3)                                                  |
| Observation Values   | (0, 255)                                                       |
| State Shape          | (720, 1280, 3)                                                 |
| State Values         | (0, 255)                                                       |


Zombies walk from the top border of the screen down to the bottom border in unpredictable paths. The agents you control are knights and archers (default 2 knights and 2 archers) that are initially positioned at the bottom border of the screen. Each agent can rotate clockwise or counter-clockwise
and move forward or backward. Each agent can also attack to kill zombies. When a knight attacks, it swings a mace in an arc in front of its current heading direction. When an archer attacks, it fires an arrow in a straight line in the direction of the archer's heading. The game ends when all
agents die (collide with a zombie) or a zombie reaches the bottom screen border. A knight is rewarded 1 point when its mace hits and kills a zombie. An archer is rewarded 1 point when one of their arrows hits and kills a zombie.
There are two possible observation types for this environment, vectorized and image-based.

#### Vectorized (Default)
Pass the argument `vector_state=True` to the environment.

The observation is an (N+1)x5 array for each agent, where `N = num_archers + num_knights + num_swords + max_arrows + max_zombies`.
> Note that `num_swords = num_knights`

The ordering of the rows of the observation look something like this:
```
[
[current agent],
[archer 1],
...,
[archer N],
[knight 1],
...
[knight M],
[sword 1],
...
[sword M],
[arrow 1],
...
[arrow max_arrows],
[zombie 1],
...
[zombie max_zombies]
]
```

In total, there will be N+1 rows. Rows with no entities will be all 0, but the ordering of the entities will not change.


**Typemasks**

There is an option to prepend a typemask to each row vector. This can be enabled by passing `use_typemasks=True` as a kwarg.

The typemask is a 6 wide vector, that looks something like this:
```
[0., 0., 0., 1., 0., 0.]
```

Each value corresponds to either
```
[zombie, archer, knight, sword, arrow, current agent]
```

If there is no entity there, the whole typemask (as well as the whole state vector) will be 0.

As a result, setting `use_typemask=True` results in the observation being a (N+1)x11 vector.

**Transformers** (Experimental)

There is an option to also pass `transformer=True` as a kwarg to the environment. This just removes all non-existent entities from the observation and state vectors. Note that this is **still experimental** as the state and observation size are no longer constant. In particular, `N` is now a
variable number.

#### Image-based
Pass the argument `vector_state=False` to the environment.

Each agent observes the environment as a square region around itself, with its own body in the center of the square. The observation is represented as a 512x512 pixel image around the agent, or in other words, a 16x16 agent sized space around the agent.

### Manual Control

Move the archer using the 'W', 'A', 'S' and 'D' keys. Shoot the Arrow using 'F' key. Rotate the archer using 'Q' and 'E' keys.
Press 'X' key to spawn a new archer.

Move the knight using the 'I', 'J', 'K' and 'L' keys. Stab the Sword using ';' key. Rotate the knight using 'U' and 'O' keys.
Press 'M' key to spawn a new knight.



### Arguments

``` python
electroplating_v1.env(
  spawn_rate=20,
  num_archers=2,
  num_knights=2,
  max_zombies=10,
  max_arrows=10,
  killable_knights=True,
  killable_archers=True,
  pad_observation=True,
  line_death=False,
  max_cycles=900,
  vector_state=True,
  use_typemasks=False,
  transformer=False,
```

`spawn_rate`:  how many cycles before a new zombie is spawned. A lower number means zombies are spawned at a higher rate.

`num_archers`:  how many archer agents initially spawn.

`num_knights`:  how many knight agents initially spawn.

`max_zombies`: maximum number of zombies that can exist at a time

`max_arrows`: maximum number of arrows that can exist at a time

`killable_knights`:  if set to False, knight agents cannot be killed by zombies.

`killable_archers`:  if set to False, archer agents cannot be killed by zombies.

`pad_observation`:  if agents are near edge of environment, their observation cannot form a 40x40 grid. If this is set to True, the observation is padded with black.

`line_death`:  if set to False, agents do not die when they touch the top or bottom border. If True, agents die as soon as they touch the top or bottom border.

`vector_state`: whether to use vectorized state, if set to `False`, an image-based observation will be provided instead.

`use_typemasks`: only relevant when `vector_state=True` is set, adds typemasks to the vectors.

`transformer`: **experimental**, only relevant when `vector_state=True` is set, removes non-existent entities in the vector state.


### Version History
* v1: Initial versions release (1.0.0)

"""
import os
import sys
import gymnasium
import numpy as np

from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle, seeding

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .manual_policy import ManualPolicy  # noqa: F401
from ..render.renderer import Renderer
from epsim.core import World,WorldObj,Slot,Crane,Actions
from epsim.core.componets import Color
#sys.dont_write_bytecode = True


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "electroplating_v1",
        "is_parallelizable": True,
        "render_fps": 60,
        "has_manual_policy": True,
    }

    def __init__(
        self,
        render_mode=None,
        vector_state=False,
        max_cycles=2000,
        args: "DictConfig"  = None,
        
    ):
        EzPickle.__init__(
            self,
            render_mode,
            vector_state,
            max_cycles
        )
        

        # whether we want RGB state or vector state
        self.vector_state = vector_state
        self.args=args
        self.world=World(args.config_directory,args.max_x)
        WorldObj.TILE_SIZE=args.tile_size
        Slot.WarningTime=args.alarm.warning
        Slot.FatalTime=args.alarm.fatal
        
        rows=args.max_x/args.cols+1+2
        self.renderer=Renderer(self.world,args.fps,rows,args.cols,args.tile_size)
        # cranes + slots + products
        # self.num_tracked = (
        #     num_archers + num_knights + max_zombies + num_knights + max_arrows
        # )
        # self.use_typemasks = True if transformer else use_typemasks
        self.typemask_width = 6
        self.vector_width = 4 + self.typemask_width #if use_typemasks else 4

        # Game Status
        self.frames = 0
        self.closed = False
        self.has_reset = False
        self.render_mode = render_mode
        self.render_on = False

        # Game Constants
        self.seed()


        # Represents agents to remove at end of cycle
        # self.kill_list = []
        # self.agent_list = []
        self.agents = []
        self.agent_name_mapping = { 'dispatch':0}
        #self.dead_agents = []

        for i in range(1,len(self.world.all_cranes)+1):
            a_name = "crane_" + str(i)
            self.agents.append(a_name)
            self.agent_name_mapping[a_name] = i



        shape = (
            [512, 512, 3]
            # if not self.vector_state
            # else [self.num_tracked + 1, self.vector_width + 1]
        )
        low = 0 #if not self.vector_state else -1.0
        high = 255 #if not self.vector_state else 1.0
        dtype = np.uint8 #if not self.vector_state else np.float64
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(low=low, high=high, shape=shape, dtype=dtype)
                    for _ in enumerate(self.agents)
                ],
            )
        )

        self.action_spaces = dict(
            zip(self.agents, [Discrete(3) if i==0 else Discrete(5) for i,_ in enumerate(self.agents)])
        )

        shape = (
            [3*args.tile_size, args.max_x*args.tile_size, 3]
            # if not self.vector_state
            # else [self.num_tracked, self.vector_width]
        )
        low = 0 if not self.vector_state else -1.0
        high = 255 if not self.vector_state else 1.0
        dtype = np.uint8 if not self.vector_state else np.float64
        self.state_space = Box(
            low=low,
            high=high,
            shape=shape,
            dtype=dtype,
        )
        self.possible_agents = self.agents
        self._agent_selector = agent_selector(self.agents)


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

 



    def observe(self, agent):
        if not self.vector_state:
            # screen = pygame.surfarray.pixels3d(self.WINDOW)

            # i = self.agent_name_mapping[agent]
            # agent_obj = self.agent_list[i]
            # agent_position = (agent_obj.rect.x, agent_obj.rect.y)

            # if not agent_obj.alive:
            #     cropped = np.zeros((512, 512, 3), dtype=np.uint8)
            # else:
            #     min_x = agent_position[0] - 256
            #     max_x = agent_position[0] + 256
            #     min_y = agent_position[1] - 256
            #     max_y = agent_position[1] + 256
            #     lower_y_bound = max(min_y, 0)
            #     upper_y_bound = min(max_y, const.SCREEN_HEIGHT)
            #     lower_x_bound = max(min_x, 0)
            #     upper_x_bound = min(max_x, const.SCREEN_WIDTH)
            #     startx = lower_x_bound - min_x
            #     starty = lower_y_bound - min_y
            #     endx = 512 + upper_x_bound - max_x
            #     endy = 512 + upper_y_bound - max_y
            #     cropped = np.zeros_like(self.observation_spaces[agent].low)
            #     cropped[startx:endx, starty:endy, :] = screen[
            #         lower_x_bound:upper_x_bound, lower_y_bound:upper_y_bound, :
            #     ]

            return None #np.swapaxes(cropped, 1, 0)

        else:
            pass
            '''
            # get the agent
            agent = self.agent_list[self.agent_name_mapping[agent]]

            # get the agent position
            agent_state = agent.vector_state
            agent_pos = np.expand_dims(agent_state[0:2], axis=0)

            # get vector state of everything
            vector_state = self.get_vector_state()
            state = vector_state[:, -4:]
            is_dead = np.sum(np.abs(state), axis=1) == 0.0
            all_ids = vector_state[:, :-4]
            all_pos = state[:, 0:2]
            all_ang = state[:, 2:4]

            # get relative positions
            rel_pos = all_pos - agent_pos

            # get norm of relative distance
            norm_pos = np.linalg.norm(rel_pos, axis=1, keepdims=True) / np.sqrt(2)

            # kill dead things
            all_ids[is_dead] *= 0
            all_ang[is_dead] *= 0
            rel_pos[is_dead] *= 0
            norm_pos[is_dead] *= 0

            # combine the typemasks, positions and angles
            state = np.concatenate([all_ids, norm_pos, rel_pos, all_ang], axis=-1)

            # get the agent state as absolute vector
            # typemask is one longer to also include norm_pos
            if self.use_typemasks:
                typemask = np.zeros(self.typemask_width + 1)
                typemask[-2] = 1.0
            else:
                typemask = np.array([0.0])
            agent_state = agent.vector_state
            agent_state = np.concatenate([typemask, agent_state], axis=0)
            agent_state = np.expand_dims(agent_state, axis=0)

            # prepend agent state to the observation
            state = np.concatenate([agent_state, state], axis=0)

            return state
            '''


    def state(self):
        """Returns an observation of the global environment."""
        if not self.vector_state:
            pass
            # state = pygame.surfarray.pixels3d(self.WINDOW).copy()
            # state = np.rot90(state, k=3)
            # state = np.fliplr(state)
        else:
            state = self.get_vector_state()

        return state

    def get_vector_state(self):
        pass
        # state = []
        # typemask = np.array([])

        # # handle agents
        # for agent_name in self.possible_agents:
        #     if agent_name not in self.dead_agents:
        #         agent = self.agent_list[self.agent_name_mapping[agent_name]]

        #         if self.use_typemasks:
        #             typemask = np.zeros(self.typemask_width)
        #             if agent.is_archer:
        #                 typemask[1] = 1.0
        #             elif agent.is_knight:
        #                 typemask[2] = 1.0

        #         vector = np.concatenate((typemask, agent.vector_state), axis=0)
        #         state.append(vector)
        #     else:
        #         if not self.transformer:
        #             state.append(np.zeros(self.vector_width))

        # # handle swords
        # for agent in self.agent_list:
        #     if agent.is_knight:
        #         for sword in agent.weapons:
        #             if self.use_typemasks:
        #                 typemask = np.zeros(self.typemask_width)
        #                 typemask[4] = 1.0

        #             vector = np.concatenate((typemask, sword.vector_state), axis=0)
        #             state.append(vector)

        # # handle empty swords
        # if not self.transformer:
        #     state.extend(
        #         repeat(
        #             np.zeros(self.vector_width),
        #             self.num_knights - self.num_active_swords,
        #         )
        #     )

        # # handle arrows
        # for agent in self.agent_list:
        #     if agent.is_archer:
        #         for arrow in agent.weapons:
        #             if self.use_typemasks:
        #                 typemask = np.zeros(self.typemask_width)
        #                 typemask[3] = 1.0

        #             vector = np.concatenate((typemask, arrow.vector_state), axis=0)
        #             state.append(vector)

        # # handle empty arrows
        # if not self.transformer:
        #     state.extend(
        #         repeat(
        #             np.zeros(self.vector_width),
        #             self.max_arrows - self.num_active_arrows,
        #         )
        #     )

        # # handle zombies
        # for zombie in self.zombie_list:
        #     if self.use_typemasks:
        #         typemask = np.zeros(self.typemask_width)
        #         typemask[0] = 1.0

        #     vector = np.concatenate((typemask, zombie.vector_state), axis=0)
        #     state.append(vector)

        # # handle empty zombies
        # if not self.transformer:
        #     state.extend(
        #         repeat(
        #             np.zeros(self.vector_width),
        #             self.max_zombies - len(self.zombie_list),
        #         )
        #     )

        # return np.stack(state, axis=0)

    def next_crane(self):
        for c in self.world.all_cranes:
            c.color=Color(255,255,255)
        self.world.next_crane()
        self.world.cur_crane.color=Color(255,0,0)
        
    def step(self, action:int):
        #agent = self.agents[self.agent_name_mapping[self.agent_selection]]
        self.world.set_command(action)
        self.world.update()
        #print(crane)
        mask=self.world.get_masks(self.world.cur_crane)
        self.next_crane()

        if self.render_mode == "human":
            self.render()
        return (0, self.world.reward, self.world.is_over, False, {"action_mask": mask})



    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        return self.renderer.render(self.render_mode)

    def close(self):
        if not self.closed:
            self.closed = True
            




    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.has_reset = True
        self.agents = self.possible_agents
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.world.reset()
        ps=[]
        for p in self.args.products:
            ps.extend([p.code]*p.num)
        self.world.add_jobs(ps)
        #self.world.cur_crane.color=Color(255,0,0)
        mask=self.world.get_masks(self.world.cur_crane)



