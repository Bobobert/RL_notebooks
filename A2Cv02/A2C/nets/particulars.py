from A2C.functions.const import *
from .base import ActorCritic

class AC_Atari(ActorCritic):
    def __init__(self, kwargs):
        super(AC_Atari, self).__init__()
        self.name = "DQN ActorCritic"
        # Defaults
        actions, lHist = kwargs.get("nActions", 4), kwargs.get("lHist", 4)
        self.config = kwargs
        self._actions, self._lh = actions, lHist
        self.rectifier = F.relu
        self.cv1 = nn.Conv2d(lHist, 32, 8, 4)
        self.cv2 = nn.Conv2d(32, 64, 4, 2)
        self.cv3 = nn.Conv2d(64, 64, 3, 1)
        # Actions
        self.fc1 = nn.Linear(3136, 256)
        self.fc2 = nn.Linear(256, actions)
        # Value function
        self.fc3 = nn.Linear(3136, 256)
        self.fc4 = nn.Linear(256, 1)

    def sharedForward(self, x):
        x = self.rectifier(self.cv1(x))
        x = self.rectifier(self.cv2(x))
        x = self.rectifier(self.cv3(x))
        return x.flatten(1)

    def valueForward(self, x):
        x = self.rectifier(self.fc3(x))
        return self.fc4(x)

    def actorForward(self, x):
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)

    def getDist(self, x):
        return distributions.Categorical(logits = x)
        
    def _new_(self):
        new = AC_Atari(self.config)
        return new

class AC_AtariHalf(ActorCritic):
    def __init__(self, kwargs):
        super(AC_AtariHalf, self).__init__()
        self.name = "DQN ActorCritic"
        # Defaults
        actions, lHist = kwargs.get("nActions", 4), kwargs.get("lHist", 4)
        self.config = kwargs
        self._actions, self._lh = actions, lHist
        self.rectifier = F.relu
        self.cv1 = nn.Conv2d(lHist, 16, 8, 4)
        self.cv2 = nn.Conv2d(16, 32, 4, 2)
        # Actions
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, actions)
        # Value function
        self.fc3 = nn.Linear(2592, 256)
        self.fc4 = nn.Linear(256, 1)

    def sharedForward(self, x):
        x = self.rectifier(self.cv1(x))
        x = self.rectifier(self.cv2(x))
        return x.flatten(1)

    def valueForward(self, x):
        x = self.rectifier(self.fc3(x))
        return self.fc4(x)

    def actorForward(self, x):
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)

    def getDist(self, x):
        return distributions.Categorical(logits = x)
        
    def _new_(self):
        new = AC_AtariHalf(self.config)
        return new

class ACNet_discrete(ActorCritic):
    def __init__(self, kwargs):
        super(ACNet_discrete, self).__init__()
        self.name = "ActorCritic"
        # Defaults
        inpts = kwargs["nInput"]
        actions, h0 = kwargs["nActions"], kwargs.get("hidden0", 52)
        self.kwargs = kwargs
        self.rectifier = F.relu
        # Actions
        self.fc1 = nn.Linear(inpts, h0)
        self.fc2 = nn.Linear(h0, actions)
        # Value function
        self.fc3 = nn.Linear(h0, 1)

    def sharedForward(self, x):
        x = self.rectifier(self.fc1(x))
        return x

    def valueForward(self, x):
        return self.fc3(x)

    def actorForward(self, x):
        return self.fc2(x)

    def getDist(self, x):
        return distributions.Categorical(logits = x)
        
    def _new_(self):
        new = ACNet_discrete(self.kwargs)
        return new