'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    
    # Make a list of states as numeric arrays to train encoder-decoder
    number_of_states_to_generate = 50000
    states = []
    counter = 0

    # Run the episodes just like OpenAI Gym
    while counter < number_of_states_to_generate:
        state = env.reset()
        done = False
        game = []
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # new part
            game.append(state)
            counter += 4
        
        states.append(game)
    env.close()
    np.save("states.npy", np.array(states, dtype="object"), allow_pickle=True)


