import os
import pickle
import zipfile
import pandas as pd
import numpy as np
import neuralflow
from neuralflow.utilities.psth import extract_psth
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from utils import select_neuron, single_unit_optimization
from utils_plot import plot_psth
import warnings
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='CPU', help='device to use')
parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs')
args = parser.parse_args()
device = args.device
max_epochs = args.max_epochs

session_id = '20211020'
neuron_idx = 188
data = pd.read_json(f'dataset_single_unit/{session_id}/neuron_{neuron_idx}.json')
data['neuron_0'] = data['neuron_0'].apply(lambda x: np.array([x]) if isinstance(x, (int, float)) else np.array(x))
data = data[data['neuron_0'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

max_epochs = max_epochs
time_offset = 0.2
single_unit_optimization(df=data, time_offset=time_offset, max_epochs=max_epochs, 
                         device=device, save_path='optimization_results_single_unit', 
                         session_id=session_id, neuron_id=neuron_idx)


