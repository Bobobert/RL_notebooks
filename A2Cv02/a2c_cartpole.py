from A2C import Agent, Learner, envMaker, config, ray
from A2C.nets import ACNet_discrete
from A2C.functions import trainSync, graphResults
from A2C.functions import NCPUS,expDir, saveConfig, Saver
from torch.utils.tensorboard import SummaryWriter
from math import ceil

EXP = "a2c"
NAME = "CartPole-v0"

envmake = envMaker(NAME)
config["envName"] = NAME
config["nInput"] = 4
config["nActions"] = 2
config["hidden0"] = 52
config["atari"] = False
config["nAgents"] = NCPUS - 1
config["n-step"] = 20
config["learningRate"] = 5e-4
config["cPolicyLoss"] = 1.0
config["cValueLoss"] = 0.5
config["entropyLoss"] = 0.0
config["nTest"] = ceil(28/(NCPUS - 1))
config["episodes_train"] = 10**5
FREQ_TEST = 5*10**3
config["freq_test"] = FREQ_TEST

path, tbpath = expDir(EXP,NAME)
saveConfig(config, path)
writer = SummaryWriter(tbpath)
saver = Saver(path)
net = ACNet_discrete

if __name__ == "__main__":
    # INIT RAY
    Agent = ray.remote(Agent)
    Learner = ray.remote(Learner)

    ray.init(num_cpus=NCPUS - 1)
    agents = [Agent.remote(envmake, net, config, 
                            seedTrain = i, 
                            seedTest = (i + 1)*(NCPUS + 1)) 
                            for i in range(NCPUS - 1)]
    learner = Learner.remote(net, config)
    # Train ActorCritic
    average_return, var_return = trainSync(agents, learner, config, saver = saver, writer = writer)
    # Save results
    graphResults(average_return, var_return, FREQ_TEST, mod = EXP, save = path)
    saver.saveAll()
    # Close tent
    ray.shutdown()
    writer.close()
