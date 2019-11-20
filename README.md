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

Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

#### Step 1: Set Up The Depedencies
To set up your python environment to run the code in this repository, follow the instructions below.
1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/jhonsonlee/dqn-navigating-in-a-virtual-world.git
cd dqn-navigating-in-a-virtual-world/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 2: Download the Unity Environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.

#### Step 3: Getting Familiar with Jupyter Notebook
The code in this repository will be written mostly in `.ipynb` file. Thus, you need to get familiar with the **jupyter notebook** which is a great editor today. Please, take your time to know Jupyter Notebook in [this blogpost](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46). 

## Instructions

### Navigating The Source Code
There are three important files which contains the source code to run and train our agent:
- `Navigation.ipynb` contains the instructions to explore the environent, train the agent to be smart and run the smart agent.
- `/src/agent.py` contains the source code which describes how the agent works
- `/src/model.py` contains the source code of deep learning model for the DQN value function

### Explore the Environment
In order to understand how to create a smart agent, we must recognize the environment first. Please, open `Navigation.ipynb` using Jupyter Notebook and follow these steps:
- Step 1 : Start the Environment
- Step 2 : Examine the State and Action Spaces
- Step 3 : Take Random Actions in the Environment

### Train the Agent
As you can see in the **Explore the Environment** section, the agent is still taking random actions and generate zero points during the gameplay. In order to create a smart agent, we need to train the agent by following these two steps:
- Step 4 : Contruct the Train Method
- Step 5 : Train the Agent

### Watch the Smart Agent Play
After we train to agent how to better play the game, we can watch the agent play with better decision and better score. Howevever, we can not see the agent play in live gameplay in this notebook. But, we can see the agent collects yellow bananas and avoids blue bananas by inspecting the score movement. Please follow the **Step 6 : Watch the Agent Play**.

## License
This repository is under the **MIT license**.
