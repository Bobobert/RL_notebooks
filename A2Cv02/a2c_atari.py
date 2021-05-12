from A2C import Agent, Learner, envMaker, config, ray
from A2C.nets import AC_AtariHalf
from A2C.functions import seeder, trainSync, graphResults
from A2C.functions import NCPUS, expDir, saveConfig, Saver
from torch.utils.tensorboard import SummaryWriter
from math import ceil

EXP = "a2c"
NAME = "Seaquest-v0"

envmake = envMaker(NAME)
config["envName"] = NAME
config["nActions"] = 6
config["atari"] = True
config["nAgents"] = NCPUS - 1
config["hidden"] = 256
config["n-step"] = 5
config["learningRate"] = 5e-4
config["cPolicyLoss"] = 1.0
config["cValueLoss"] = 0.2
config["entropyLoss"] = 0.01
config["nTest"] = ceil(28/(NCPUS - 1))
config["episodes_train"] = 10**5
FREQ_TEST = 10**3
config["freq_test"] = FREQ_TEST

path, tbpath = expDir(EXP,NAME)
saveConfig(config, path)
seeder(8088)
writer = SummaryWriter(tbpath)
saver = Saver(path)
net = AC_AtariHalf

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
