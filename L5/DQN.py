import sys
import os
import numpy as np

# Add the parent directory (CS230) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from L4
from L4.Activation_funcs import ActivationFunctions
from L4.Evaluation import Evaluation
from L4.Initialization import Initialization


class QNN:
    def __init__(self):
        pass


print("import successfully !!! ")