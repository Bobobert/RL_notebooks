"""
    RobBob's bag of tools
"""
import os
import sys
import time
from torch import save, load, device
from .vars import Variable
import pickle
import json
import re


def timeFormated() -> str:
    return time.strftime("%H-%M_%d-%b-%y", time.gmtime())

def timeFormatedS() -> str:
    #return time.strftime("%Y-%B-%d-_%H-%M-%S", time.gmtime())
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

def timeDiffFormated(start):
    tock = time.time()
    total = tock - start
    hours = int(total//3600)
    mins = int(total%3600//60)
    secs = int(total%3600%60//1)
    if hours > 0:
        s = "{}h {}m {}s".format(hours, mins, secs)
    elif mins > 0:
        s = "{}m {}s".format(mins, secs)
    elif secs > 0:
        s = "{}s".format(secs)
    else:
        s = "< 0s"
    return s, tock

def goToDir(path):
    home = os.getenv('HOME')
    try:
        os.chdir(os.path.join(home, path))
    except:
        os.chdir(home)
        os.makedirs(path)
        os.chdir(path)
    return os.getcwd()

def createFolder(path:str, mod:str): # Deprecated
    start = mod + '_' + timeFormated()
    new_dir = os.path.join(path, start)
    new_dir = goToDir(new_dir)
    return start, new_dir

def tbDir(expName:str) -> str:
    """
        Returns the default tensorboard direction
        given the experiment key.
    """
    return genDir(expName, "tensorboard")

def expDir(expName:str, envName:str) -> (str, str):
    """
        Returns the default folders for the experiment
        with the environment name description.

        returns
        -------
        expdir
    """
    t = timeFormatedS()
    return genDir(expName, envName, t), genDir(expName, envName, "tensorboard", t)

def genDir(*args) -> str:
    dr = os.getenv('HOME')
    adds = ["rl_results", *args]
    for s in adds:
        dr = os.path.join(dr, s)
    os.makedirs(dr, exist_ok=True)
    return dr

def saveConfig(config:dict, expDir:str):
    """
        Dumps the config dictionary into a 
        json file.
    """
    fh = open(expDir + "/config.json", "w")

    def default(o):
        if isinstance(o, Variable):
            return o.__repr__()

    json.dump(config, fh, indent=4, default = default)
    fh.close()

def loadConfig(expDir):
    fh = open(expDir + "/config.json", "r")
    config = json.load(fh)
    fh.close()
    return config

def timeToStop(results, expected = None):
    tock = time.time()
    diff = tock - results["time_start"]
    results["time_elapsed"] = diff
    results["time_execution"] += [timeFormatedS()]
    stop = False
    if expected is not None:
        stop = True if (diff // 60) >= expected else False
    return results, stop

class Tocker:
    def __init__(self):
        self.tick
    @property
    def tick(self):
        self.start = time.time()
        return self.start
    @property
    def tock(self):
        s, self.start = timeDiffFormated(self.start)
        return s
    @property
    def tocktock(self):
        """
        Returns the time elapsed since the last tick in minutes
        """
        return (time.time() - self.start) * 0.016666667
    def lockHz(self, Hz:int):
        tHz = 1 / Hz
        remaind = time.time() - self.start
        remaind = tHz - remaind
        if remaind > 0:
            time.sleep(remaind)
            return True

class Stack:
    """
    Dict stack working in a FIFO manner
    """
    def __init__(self):
        self.stack = dict()
        self.min = 0
        self.actual = 0
    def add(self, obj):
        self.stack[self.actual] = obj
        self.actual += 1
    def pop(self):
        poped = self.stack[self.min]
        self.stack.pop(self.min)
        self.min += 1
        return poped
    def __len__(self):
        return len(self.stack)

class Reference:
    _loaded_ = False
    def __init__(self, obj, 
                        name: str,
                        limit:int,
                        torchType:bool = False,
                        device = device("cpu"),
                        loadOnly:bool = True):
        self.torchType = torchType
        self.name = name
        self.ref = obj
        self.prevVersions = Stack()
        self.limit = limit
        self.device = device
        self._version = 0
        self._LO_ = loadOnly
    
    def save(self, path):
        if self._LO_:
            None
        if self.torchType:
            self.saveTorch(path)
        else:
            self.savePy(path)
        self.clean(path)

    def clean(self, path):
        if len(self.prevVersions) >= self.limit:
            target = self.prevVersions.pop()
            #target = os.path.join(path, target)
            os.remove(target)
    
    @staticmethod
    def loaderAssist(path):
        os.chdir(path)
        files = os.listdir()
        print("Files on direction:")
        for n, File in enumerate(files):
            print("{} : {}".format(n, File))
        while 1:
            choice = input("Enter the number for the file to load :")
            choice = int(choice)
            if choice > len(files) or not isinstance(choice, int) or choice < 0:
                print("Number not valid. Please try again.")
            else:
                break
        return os.path.join(path, files[choice])

    def load(self, path):
        self._loaded_ = True
        print("Trying to load in object {}".format(self.name))
        target = self.loaderAssist(path)
        self._version = int(re.findall("_v\d+", target)[0][2:]) + 1
        if self.torchType:
            self.loadTorch(target, self.device)
        else:
            self.loadObj(target)
    
    def loadTorch(self, path, device):
        model = load(path, map_location=device)
        self.ref.load_state_dict(model, strict = True)
        print("Model successfully loaded from ", path)
        
    def loadObj(self, path):
        fileHandler = open(path, 'rb')
        self.ref = pickle.load(fileHandler)
        fileHandler.close()
        print("Object successfully loaded from ", path)

    def saveTorch(self, path):
        name = self._gen_name() + ".modelst"
        path = os.path.join(path, name)
        try:
            stateDict = self.ref.state_dict()
            save(stateDict, path)
            self.prevVersions.add(path)
        except:
            None

    def savePy(self, path):
        name = self._gen_name() + ".pyobj"
        path = os.path.join(path, name)
        if sys.getsizeof(self.ref) < LIMIT_4G:
            fileHandler = open(path, "wb")
            pickle.dump(self.ref, fileHandler)
            fileHandler.close()
            self.prevVersions.add(path)

    def _gen_name(self):
        self._version += 1
        return self.name + "_v{}".format(self._version) + "_" + timeFormated()

class Saver():
    """
    Object that administrates objects to dump
    save files if possible.

    parameters
    ----------
    envName: str

    path: str
        Path relative to Home to dump the saved files
    """
    def __init__(self, path:str,
                    limitTimes:int = 10,
                    saveFreq:int = 30):
        
        self.dir = path
        self._objRefs_ = []
        self.names = set()
        self.limit = limitTimes
        self.time = Tocker()
        self.freq = saveFreq

    def start(self):
        self.time.tick

    def check(self):
        if self.time.tocktock >= self.freq:
            self.saveAll()
            self.time.tick

    def addObj(self, obj, 
                objName:str,
                isTorch:bool = False,
                device = device("cpu"),
                loadOnly:bool = False):

        if objName in self.names:
            raise KeyError
        self.names.add(objName)
        self._objRefs_ += [Reference(obj, 
                                    objName, 
                                    self.limit,
                                    isTorch,
                                    device,
                                    loadOnly)]
    
    def saveAll(self):
        for ref in self._objRefs_:
            ref.save(self.dir)

    def load(self, path):
        for ref in self._objRefs_:
            ref.load(path)
