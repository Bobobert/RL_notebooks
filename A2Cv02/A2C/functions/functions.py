"""
    Custon functions for the program
"""
from .const import *

def envMaker(name):
    def ENV(seed = None):
        if seed is not None and seed < 0:
            seed = None
        env = make(name)
        seeds = env.seed(seed)
        return env, seeds

    return ENV

def getDevice(cudaTry:bool = True):
    if torch.cuda.is_available() and cudaTry:
        return torch.device("cuda")
    return DEVICE_DEFT

# Custom functions
def toT(arr:np.ndarray, device = DEVICE_DEFT, dtype = F_DTYPE_DEFT, grad: bool = False):
        arr = np.squeeze(arr) # Fix due to a problem in Pendulum Observation, seems arbritary when it fails or not.
        return torch.as_tensor(arr, dtype = dtype, device = device).unsqueeze(0).requires_grad_(grad)

def copyDictState(net, grad:bool = True):
    newSD = dict()
    sd = net.state_dict()
    for i in sd.keys():
        t = sd[i]
        newSD[i] = t.new_empty(t.shape, requires_grad=grad).copy_(t)
    return newSD

def cloneNet(net):
    new = net._new_()
    new.load_state_dict(copyDictState(net), strict = True)
    return new.to(net.device)

def updateNet(net, targetLoad):
    #net.opParams = copyStateDict(net)
    if isinstance(targetLoad, dict):
        net.load_state_dict(targetLoad)
    elif isinstance(targetLoad, list):
        for p, pt in zip(targetLoad, net.parameters()):
            pt.requires_grad_(False) # This is a must to change the values properly
            pt.copy_(p).detach_()
            pt.requires_grad_(True)

def getDictState(net, cpu):
    stateDict = net.state_dict()
    if cpu:
        for key in stateDict.keys():
            stateDict[key] = stateDict[key].to(DEVICE_DEFT)
    return stateDict

def getListState(net, cpu):
    params = []
    for p in net.parameters():
        params += [p if not cpu else p.clone().to(DEVICE_DEFT)]
    return params

def graphResults(means, variance, testFreq, mod:str, dpi=200, save = ""):
    mean = np.array(means)
    stds = np.sqrt(np.array(variance))
    fig = plt.figure(dpi=dpi)
    plt.title("{}: Accumulated Reward per Episode".format(mod))
    plt.xlabel("Episode")
    plt.ylabel("Accumulate Reward")
    x = np.arange(0,len(means)*testFreq, testFreq)
    plt.plot(x, means, label = "Accumulate Reward", lw = 2)
    plt.fill_between(x, means - stds, means + stds, alpha = 0.1)

    if save != "":
        plt.savefig(save + "/graph.png", dpi = dpi)