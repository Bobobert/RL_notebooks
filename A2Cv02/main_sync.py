from A2C import Agent, Learner, envMaker, config, ray
from A2C.nets import ACNet_discrete
from A2C.functions import train, trainSync, graphResults
from A2C.functions import NCPUS, linearSchedule, expDir, saveConfig, Saver
from copy import deepcopy
from math import ceil

EXP = "a2c"
NAME = "LunarLander-v2"

envmake = envMaker(NAME)
config["envName"] = NAME
config["nInput"] = 8
config["nActions"] = 4
config["hidden0"] = 52
config["atari"] = False
config["nAgents"] = NCPUS - 1
config["n-step"] = 20
config["learningRate"] = 5e-4
config["cPolicyLoss"] = 1.0
config["cValueLoss"] = 0.5
config["entropyLoss"] = 0.0
config["nTest"] = ceil(28/(NCPUS - 1))
config["episodes_train"] = 10*10**3 + 1
FREQ_TEST = 500
config["freq_test"] = FREQ_TEST

path, _ = expDir(EXP,NAME)
saveConfig(config, path)
saver = Saver(path)

if __name__ == "__main__":
    # INIT RAY
    Agent = ray.remote(Agent)
    Learner = ray.remote(Learner)

    ray.init(num_cpus=NCPUS - 1)
    agents = [Agent.remote(envmake, ACNet_discrete, deepcopy(config), 
                            seedTrain = i, 
                            seedTest = (i + 1)*(NCPUS + 1)) 
                            for i in range(NCPUS - 1)]
    learner = Learner.remote(ACNet_discrete, deepcopy(config))
    # Train ActorCritic
    average_return, var_return = trainSync(agents, learner, config, saver = saver)
    # Save results
    graphResults(average_return, var_return, FREQ_TEST, mod = EXP, save = path)
    saver.saveAll()
    # Close tent
    ray.shutdown()
