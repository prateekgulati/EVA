## Deep Q - Reinforcement Learning
 ### Goal
 Build a simulation for a city where an ambulance picks and drops the patients to the city hospital

### Files
 - map.py: The main file where the app is running. This includes setting up the environment, objects configuration, car movement, sensor readings, rewards and canvas interaction including drawing obstruction and saving/loading brain.
- ai.py: This file includes the deep Q network which has deep learning network to predict the action, a memory which stores the recent experience, a rolling window for recent rewards, along with training and updates
- car.kv: This is a kivy file that includes the graphical configuration of all the objects that are used in the kivy app. 

### Environment
 - CityMap: This is a map of the city used as the reference for all states and actions
 - Mask: A mask file is created based on the road network in the citymap where the car should move. 
 -  Sand: A sand variable is derived from the mask. 
     - NoSand: This is basically a road and the sand value is 0. The car can achieve max velocity and reward in this region.
     - SomeSand: At the intersection of roads where sand value is between 0.1 and 0.5. The car slows down to avoid skidding and bumping into sand/edges. The reward is generally high.
     - HighSand: Everywhere else, the sand value is high. These areas have the highest obstruction, least velocity, and high penalty.
### Action
There are three possible actions: 0, 1, and 2 representing the direction (left, straight and right) in which the car will move respectively.

### Rewards
Depending on the state of the car and which action it takes, there is a reward that it receives. There are 4 possible rewards:
- Reward for moving on the sand: -3
- Reward for moving on the road: -1
- Reward for moving on the road and towards the direction of destination: -0.1
- Reward for bumping into boundary -10

### Network
The deep learning model has 1 input layer, 2 dense layers each combined with ReLU activation and an output layer. 
The input layer has 5 inputs: 
 - 3: One from each sensor - the value of sand around them
 - 2: Value of angle between car and x-axis - both clockwise and anticlockwise
The first FC layer has 30 output and the second FC has 50 output. The output layer gives 3 outputs corresponding to each action.

### Experience Replay
A memory that stores past experiences. The size of this memory is 100000. Once the memory is full, the old experiences are deleted as the new ones get added.

### Rolling Window
A rolling window that stores the rewards for action taken. The size of the window is 1000. Once the memory is full, the old experiences are deleted as the new ones get added.

## Simulation
#### Youtube Video:  

[![Watch the video](https://img.youtube.com/vi/kjkRsZe4TDc/hqdefault.jpg)](https://www.youtube.com/watch?v=kjkRsZe4TDc)
