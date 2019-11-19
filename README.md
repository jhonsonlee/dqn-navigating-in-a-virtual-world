[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Design An RL Agent to Collect Fresh Bananas

## Introduction

The goal of this project is to create an agent to play a game in a virtual world. An agent here can be seen as a player, but the player is the computer which use artificial intelligence. The virtual world are full of two types of bananas: the fresh one(yellow) and the rotten one(blue). The agent need to collect the yellow bananas as many as possible and avoid the blue one. 

The virtual world can be seen in the following **.gif file**.

![Trained Agent][image1]

In order to successfully create a smart agent, we use Deep Reinforcement Learning(DRL) algorithm which is one well-known method in Artificial Intelligent. DRL algorithm allow the agent, in our case the player, learn how to play the while interacting in the virtual world. In other words, the agent starts the game without any knowledge, learns slowly by interacting the game and finally masters the game after enough interaction in the virtual world.

Moreover, we use the Deep Q-Networks(DQN) inspired by [the paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) which described successful implementation of Reinforcement Learning algorithm to play in various Atari games.

You are encouraged to learn the concept of Reinforcement Learning and [the DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and use this repo to better understand the implementation of an DQN agent to play the banana game.

## The Details on The Environment and The Agent
#### The Environment
In reinforcement learning, the virtual world where the agent(the player) play is usually called **the environment**. Thus, the word **environment** refers to the square world full of bananas in our case.

The environment is a virtual world full of bananas explained in the introduction section. The environment is further processed to a simple representation of **state space**. The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction

#### The Agent
The agent is the player here. The agent will get a reward of +1 for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

## Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
