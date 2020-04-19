
## TD3 - Reinforcement Learning

### Goal
 Build a simulation of a city where the car moves from origin to target using TD3

> Note: The current submission includes two parts:
> 1) Workable code: This is simulation integrated with TD3 but the states include reading from car sensors. This code works fine and video shared below is recorded on this code
> 2) In progress: This is simulation integration with TD3 which takes cropped image around the car and uses that to take an action. This code has few errors and is in progress

### Environment
 - CityMap: This is a map of the city used as the reference for all states and actions
 - Mask: A mask file is created based on the road network in the citymap where the car should move. 
 -  Sand: A sand variable is derived from the mask. 
     - NoSand: This is the road and the sand value is 0 here. The car can achieve max velocity and reward in this region.
     - HighSand: Everywhere else, the sand value is high. These areas have the highest obstruction, least velocity, and high penalty.

### Episode
The sequence of movement of the car till it reaches its target is marked by an episode. An episode is completed when
- Car reaches target
-  The Total reward for the car is below -2000
On completion of each episode, the car starts center of the map

### Action
There is one action for now, rotation. This is a continuous variable in the range of [-8,8]. This tells the car by which angle it should rotate based on the given state.

### Rewards
Depending on the state of the car and which action it takes, there is a reward that it receives. There are 4 possible rewards:
- Reward for moving on the sand: -10
- Reward for moving on the road: -0.2
- Reward for moving on the road and towards the direction of destination: +5
- Reward for bumping into boundary -10


### Network
Reinforcement learning network is based on the actor-critic model. 


## Simulation
#### Youtube Video:  


![Watch the video](https://img.youtube.com/vi/lNEzFhPn-5g/hqdefault.jpg)(https://www.youtube.com/watch?v=lNEzFhPn-5g)

### ToDo:
- Add triangle and replace it with car position and orientation
- Fine-tune CNN designed for Actor and Critic
- Resize image from 64x64 to 28x28
- Start with negative initialization of total reward and keep on adding positive rewards instead of giving penalties
- Include car orientation with axis formed by the target and car's current position and distance from the target in states. This will need to build model api where CNN part will work on images, and output from gap will be added to FC layer along with two other states variables. These together will predict the action.
- Add more targets so that the model doesn't overfit and memorize the path.
- Work and improve the simulation user experience by adding a few graphics objects 
- Adding velocity/acceleration as another action, so the car can automatically speed up and down based on the state.
