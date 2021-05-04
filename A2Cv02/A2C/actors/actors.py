from A2C.functions.const import *
from A2C.functions.functions import updateNet, getListState, toT
from A2C.functions import Variable

class Agent:
    def __init__(self, envMaker,
                    actorCritic,
                    config,
                    seedTrain: int = -1,
                    seedTest: int = -1):

        self.env, self.envSeed = envMaker(seedTrain)
        self.envTest, self.envTestSeed = envMaker(seedTest)
        self.config = config
        self.AC = actorCritic(config)
        self._lastkey = 0

        self.nStep = config.get("n-step", 1)
        self.done = True
        self.obs = None
        self.steps = 0
        self.lastUpdate = 0
        self.episodes = 0

        self.atari = config["atari"]
        self.frame = None

    def updateAC(self,targetParams, key):
        if key <= self._lastkey:
            return None
        updateNet(self.AC, targetParams)
        self._lastkey = key

    def atariProcessState(self, frame):
        if self.frame is None:
            self.frame = np.zeros([self.config.get("lHist", 4), 84, 84], dtype = np.float32)
        else:
            self.frame = np.roll(self.frame, 1, 0)
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:,:,0], (84, 84)) / 255
        self.frame[0] = frame
        return toT(self.frame)

    def processObs(self, obs):
        if self.atari:
            return self.atariProcessState(obs)
        return toT(obs)

    def developEpisode(self):
        env = self.env
        obs = self.obs

        if self.done:
            obs = env.reset()
            self.frame = None
            obs = self.processObs(obs)
            self.done = False

        log_actions, entropies, values, rewards = [], [], [], List()

        while True:
            # Get output from actor-critic
            valueOut, policyOut = self.AC.forward(obs)
            # Sample action from the actions distribution
            action, log_action, entropy = self.AC.sampleAction(policyOut)
            # Observe next state and reward
            nextObs, reward, done, _ = env.step(action)
            # Register experience
            self.steps += 1
            values += [valueOut]
            log_actions += [log_action]
            entropies += [entropy]
            rewards.append(float(reward))

            obs = self.processObs(nextObs)

            if (self.steps - self.lastUpdate) == self.nStep or done:
                # set variables
                self.done = done
                self.episodes += 1 if done else 0
                self.frame = None if done else self.frame
                self.obs = obs
                self.lastUpdate = self.steps
                # prepare bootstrapping
                rewards.append(0.0 if done else self.AC.getValue(obs))
                break
        # Calculate grads
        self.calculateGradients(log_actions, values, rewards, entropies)       
        return self.gradients, 1 if self.done else 0
    
    def calculateGradients(self, log_actions, baselines, rewards, entropies):
        # Set tensors
        n = len(log_actions)
        log_actions = torch.cat(log_actions)
        baselines = torch.cat(baselines).squeeze(1)
        entropies = torch.cat(entropies)
        returns = self.calculateReturns(rewards, self.config.get("gamma", 1.0))
        returns = torch.as_tensor(returns, dtype=F_DTYPE_DEFT)
        # Get parameters
        beta = self.config.get("entropyLoss", 0.0)
        cPolicyLoss = config.get("cPolicyLoss", 1.0)
        cValueLoss = config.get("cValueLoss", 1.0)
        #gradClip = config.get("gradClip", np.inf)
        # Reset grads
        self.AC.zero_grad()
        # Calculate losses
        policyTarget = returns - baselines.detach()
        lossPolicy = - 1.0 * torch.mean(log_actions * policyTarget.unsqueeze(1) + beta * entropies)
        lossValue = F.mse_loss(baselines, returns)
        loss = cPolicyLoss * lossPolicy + cValueLoss * lossValue
        loss.backward()
        # Clip the gradients
        #torch.nn.utils.clip_grad_norm_(self.AC.parameters(), gradClip)

    @property
    def gradients(self):
        grads = []
        for p in self.AC.parameters():
            grads += [p.grad.clone().detach_()]
        return grads

    @staticmethod
    @njit
    def calculateReturns(rewards:List, gamma:float):
        n = len(rewards) - 1
        returns = np.zeros(n, dtype = np.float32)
        # Bootstrapping
        returns[n - 1] = rewards[n - 1] + gamma * rewards[n]
        for i in range(n - 2, -1, -1):
            returns[i] = rewards[i] + gamma * returns[i + 1]
        return returns

    def test(self, prnt: bool = True):
        env = self.envTest
        # Reset the atari observation if needed
        lastFrame = self.frame
        # Setting variables
        nTest, stepsTest = self.config["nTest"], self.config["stepsTest"]
        meanRunReward, meanC, stepsMean, var = 0, 1 / nTest, 0, []
        # Running tests
        for i in range(nTest):
            # Start Test
            done, accReward, steps = False, 0.0, 0

            obs = env.reset()
            self.frame = None
            obs = self.processObs(obs)

            while not done:
                action, _, _ = self.AC.getAction(obs)
                nextObs, reward, done, _ = env.step(action)
                accReward += reward
                steps += 1
                if stepsTest > 0 and steps > stepsTest:
                    done = True
                else:
                    obs = self.processObs(nextObs)
            # Save results from test
            #if isinstance(accReward, (np.ndarray)):
             #   accReward = accReward[0]
            meanRunReward += accReward * meanC
            stepsMean += steps * meanC
            var += [accReward]
        # Calculate variance
        tVar = 0
        for v in var:
            tVar += meanC * (v - meanRunReward)**2
        if prnt:
            s = "Means: accumulate_reward {:.3f}, variance {:.3f}, steps {:.3f}".format(meanRunReward, tVar, stepsMean)
            print(s)
        # Reseting state
        self.frame = lastFrame
        return meanRunReward, tVar, stepsMean


class Learner:
    def __init__(self, actorCritic,
                        config):

        self.AC = actorCritic(config)
        self.config = config
        self.updates = 0

        optimizer = config.get("optimizer", "Adam")
        params = self.AC.parameters()
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=config["learningRate"], **config["optimizerArgs"])
        elif optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(params, lr=config["learningRate"], **config["optimizerArgs"])
        else:
            raise KeyError("Optimizer key not recognized")
        
        self.zeroGrad()

    def zeroGrad(self):
        """
        Sets the gradient from NoneType or else to zero.

        returns None
        """
        for p in self.AC.parameters():
            p.grad = p.new_zeros(p.shape)

    def addGrads(self, *grads):
        """
        Gets all the gradients from the other nets and accumulates in the main network.

        """
        for grad in grads:
            for p, g in zip(self.AC.parameters(), grad):
                p.grad.add_(g)
    
    def optimize(self):
        """
        Applies the optimizer step and set to zero the gradient.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.updates += 1
        return self.parameters, self.updates

    @property
    def parameters(self):
        """
        Returns the cpu list of parameters
        """
        return getListState(self.AC, True)

    def model(self):
        return self.AC
