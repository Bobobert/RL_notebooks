"""
    Base function for the actor critic
"""

from A2C.functions.const import *

class ActorCritic(nn.Module):
    """
    Class designs to host both actor and critic for those architectures when a start 
    part is shared like a feature extraction from a CNN as for DQN-Atari.
    """
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.name = "Actor_critic"
        self.discrete = True
        self.__dvc__ = None
    
    def sharedForward(self, x):
        """
        From the observation, extracts the features. Recomended to return the 
        flatten tensor in batch form
        """
        raise NotImplementedError

    def valueForward(self, x):
        """
        From the feature extraction. Calculates the value from said observation
        """
        raise NotImplementedError

    def actorForward(self, x):
        """
        From the feature extraction. Calculates the raw output to represent the parameters
        for the actions distribution.
        """
        raise NotImplementedError

    def getDist(self, x):
        """
        From the actorForward, returns the corresponding pytorch distributions objecto to 
        sample the action from and to return .log_prob()
        """
        raise NotImplementedError

    def forward(self, x):
        features = self.sharedForward(x)
        values = self.valueForward(features)
        raw_actor = self.actorForward(features.clone())

        return values, raw_actor

    def getAction(self, x):
        """
        From a tensor observation returns the sampled actions and 
        their corresponding log_probs from the distribution.

        returns
        -------
        action, log_prob, entropy
        """
        with torch.no_grad():
            distParams = self.actorForward(self.sharedForward(x))
        return self.sampleAction(distParams)
    
    def sampleAction(self, params):
        """
        Creates, samples and returns the action and log_prob for it

        parameters
        ----------
        params:
            Raw output logits from the network

        returns
        action, log_prob, entropy
        """
        dist = self.getDist(params)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        if self.discrete:
            action = action.item()
        else:
            action.to(DEVICE_DEFT).squeeze(0).numpy()

        return action, log_prob, entropy

    def getValue(self, x):
        """
        Form a tensor observation returns the value approximation 
        for it with no_grad operation.
        """
        with torch.no_grad():
            value = self.valueForward(self.sharedForward(x))
        return value.item()

    def _new_(self):
        """
        This method must return the same architecture when called from a 
        already given network.
        """
        raise NotImplementedError

    @property
    def device(self):
        if self.__dvc__ is None:
            self.__dvc__ =  next(self.parameters()).device
        return self.__dvc__