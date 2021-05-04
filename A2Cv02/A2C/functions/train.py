from .const import *
from .functions import graphResults
import time

def train(actor, learner, config, saver = None):
    """
    A2C function to train one agent and learner without ray,
    this is in synchronous fashion.
    """
    episodes, testFreq = config["episodes_train"], config["freq_test"]
    testAR, testRV, testS = [], [], []
    totEpisodes, episode = 0,0
    while episodes > totEpisodes:
        # Develop Episode in actor
        grad, episode = actor.developEpisode()
        # If a episode was finished
        if episode:
            totEpisodes += episode
            print("Completed episodes {} of {}".format(totEpisodes, episodes))
            # Testing
            if totEpisodes % testFreq == 0:
                mean, var, steps = actor.test()
                testAR += [mean]
                testRV += [var]
                testS += [steps]
        # Add grads, optimize and sync
        learner.addGrads(grad)
        newParams, paramsKey = learner.optimize()
        actor.updateAC(newParams, paramsKey)
    
    return testAR, testRV


def trainSync(actors, learner, config, saver = None):
    """
        A2C function to train agents initialized with ray, 
        in a synchronous fashion.

        returns
        -------
        lists
        meanAccumulatedReward, varianceAccumulatedReward, steps

    """
    # Test administration variables
    episodes, freqTest = config["episodes_train"], config["freq_test"]
    testAR, testRV, testS = [], [], []
    totEpisodes, lastTot, lastTest = 0, 0, 0
    startTime, totTime = time.time(), 0

    if saver is not None:
        model = ray.get(learner.model.remote())
        saver.addObj(model, "AC_model", True)

    while totEpisodes < episodes:
        # Testing perfomance
        if (totEpisodes - lastTest) >= freqTest:
            print("Starting testing", end="")
            results = ray.get([actor.test.remote(False) for actor in actors])
            ein, zwei, drei = 0,0,0
            # Upack results
            for i, result in enumerate(results):
                mean, var, steps = result
                oldEin = ein
                ein += (mean - ein) / (i + 1)
                zwei += (ein - oldEin) * (mean - ein)  
                drei += (steps - drei) / (i + 1)
            testAR += [ein]
            testRV += [zwei]
            testS += [drei]
            lastTest = totEpisodes
            print(" -- Test Completed, mean accumulated Reward {:.2f}".format(ein))
        # Start running actors
        runningActors = [actor.developEpisode.remote() for actor in actors]
        # Obtain grads from agents
        for i in range(len(runningActors)):
            [ready], runningActors = ray.wait(runningActors)
            grad, episode = ray.get(ready)
            totEpisodes += episode
            ray.get(learner.addGrads.remote(grad))
        if totEpisodes != lastTot:
            elapsedTime = time.time() - startTime
            startTime = time.time()
            totTime += elapsedTime
            print("Completed episodes {} of {} in {:d}m:{:.1f}s".format(totEpisodes, episodes, int(elapsedTime) // 60, elapsedTime % 60))
            lastTot = totEpisodes
        # Calculate and Sync latest parameters
        newParams, key = ray.get(learner.optimize.remote())
        # Sync params
        ray.get([actor.updateAC.remote(newParams, key) for actor in actors])
        if saver is not None:
            saver.check()
            
    return testAR, testRV


def trainAsync(actors, learner, config, saver = None):
    """
        A3C function to train agents initialized with ray, 
        in a A2C asynchronous fashion.

        returns
        -------
        lists
        meanAccumulatedReward, varianceAccumulatedReward, steps

    """
    # TODO Complete the dict for asynchronous update
    episodes, freqTest = config["episodes_train"], config["freq_test"]
    testAR, testRV, testS = [], [], []
    totEpisodes, lastTot, lastTest = 0, 0, 0

    directory = dict()
    # actor : ref to ray actor
    # workName: another type of key (?)
    # workRef: store in here the ref from the ray.wait() worker
    # while still working, collect if when ready
    for i, actor in enumerate(actors):
        directory[i] = {"actor": actor,
                        "workName": None,
                        "workRef": None}

    runnningLearn, runningTest = None, None
    while totEpisodes < episodes:
        # Testing perfomance
        if (totEpisodes - lastTest) >= freqTest:
            runningTest = actors[0].test.remote()
            runningActors = [actor.developEpisode.remote() for actor in actors[1:]]
            lastTest = totEpisodes
        else:
            runningActors = [actor.developEpisode.remote() for actor in actors]
        # Obtain grads from agents
        for i in range(len(runningActors)):
            [ready], runningActors = ray.wait(runningActors)
            grad, episode = ray.get(ready)
            totEpisodes += episode
            ray.get(learner.addGrads.remote(grad))
        if totEpisodes != lastTot:
                print("Completed episodes {} of {}".format(totEpisodes, episodes))
                lastTot = totEpisodes
        # Calculate and Sync latest parameters
        newParams, key = ray.get(learner.optimize.remote())
        # If actor in testing, get results
        if runningTest is not None:
            mean, var, steps = ray.get(runningTest)
            testAR += [mean]
            testRV += [var]
            testS += [steps]
            runningTest = None
        # Sync params
        ray.get([actor.updateAC.remote(newParams, key) for actor in actors])
    
    return testAR, testRV
