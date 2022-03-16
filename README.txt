--- README TEAM AI ---

Here you can find the code submitted by TEAM AI for the "Praktikum aus KI: 2021/2022" at TU Darmstadt.

TEAMAI: Christina Behm, Johannes Gaese, Tobias Gockel, Wanja de Sombre, Lorenz Thiede

The code is structured in four github repositories:

A) Agent007
GitHub: https://github.com/wanjads/agent007
Description: In this repository, you can find the implementation of Agent007, a non-learning improved SimpleAgent. The repository is based on Resnick's pommerman playground and is used analogously. To start a game, use the file examples/simple_ffa_run.py. Our agent can be found in the file pommerman/agents/agent007.py

B) PommerLearn
GitHub: https://github.com/lthiede/PommerLearn

C) pommerman_il
GitHub: https://github.com/wanjads/pommerman_il
Description: This repository provides the weight initalization approaches. 
For using the DodgeBoard or the BombBoard you have to set the pommerman enviroment accordingly.
To start a game use the main.py file and set the pommerman enviorment in line 44 to the enviorment you want to use.
BombBoard: pommerman.make('BombBoard-v0', self.agent_list)
DodgeBoard: pommerman.make('DodgeBoard-v0', self.agent_list)

You can change the default constants of these boards in the constants.py
We added make_bomb_board function in the v0.py and utility.py and two functions to define the boards in the configs.py.
Also we added a stoner_agent.py.

D) PommermanAI 
GitHub: https://github.com/JohnnyG502/pommermanAI
