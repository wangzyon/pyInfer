import os
from os.path import dirname as dname

# SET ENV
os.environ["RootFolder"] = dname(dname(__file__))
os.environ["WorkspaceFolder"] = f"{os.environ['RootFolder']}/workspace"
os.environ["AppFolder"] = f"{os.environ['RootFolder']}/applications"