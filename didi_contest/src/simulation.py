# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import time

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo-gui"
sumoConfig = "/home/nlp/bigsur/devel/didi/sumo/didi_contest/di.sumo.cfg"
import traci

def list_of_n_phases(TLIds):
    n_phases = []
    for light in TLIds:
        n_phases.append(int((len(traci.trafficlights.getRedYellowGreenState(light)) ** 0.5) * 2))
    return n_phases

def makemap(TLIds):
    maptlactions = []
    n_phases = list_of_n_phases(TLIds)
    for n_phase in n_phases:
        mapTemp = []
        if len(maptlactions) == 0:
            for i in range(n_phase):
                if i%2 == 0:
                    maptlactions.append([i])
        else:
            for state in maptlactions:
                for i in range(n_phase):
                    if i%2 == 0:
                        mapTemp.append(state+[i])
            maptlactions = mapTemp
    return maptlactions

def main():
    # Control code here
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--start"]
    print(sumoCmd)
    traci.start(sumoCmd)
    TLIds = traci.trafficlights.getIDList()
    actionsMap = makemap(TLIds)
    detectorIDs = traci.inductionloop.getIDList()
    state_space_size = traci.inductionloop.getIDCount()*2
    action_space_size = len(actionsMap)
    #agent = Learner(state_space_size, action_space_size, 0.0)
    #agent.load("./save/traffic.h5")
    # Get number of induction loops
    #state = get_state(detectorIDs)
    total_reward = 0
    simulationSteps = 0
    while simulationSteps < 1000:
        #action = agent.act(state)
        lightsPhase = actionsMap[0]
        for light, index in zip(TLIds, range(len(TLIds))):
            traci.trafficlights.setPhase(light, lightsPhase[index])
        for i in range(2):
            traci.simulationStep()
            time.sleep(0.4)
        simulationSteps += 2
        #next_state = get_state(detectorIDs)
        #reward = calc_reward(state, next_state)
        #total_reward += reward
        #agent.remember(state, action, reward, next_state)
        #state = next_state
    traci.close()
    print "Simulation Reward: {}".format(total_reward)

if __name__ == '__main__':
    main()