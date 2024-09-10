import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__)))

import metrics
from metrics import compute_metrics
from protein_task import ProteinTask, get_feature_tensor, get_protein_task
