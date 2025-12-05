import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from utils import single_unit_optimization
import argparse
from tqdm import tqdm
"""
This script is used to optimize all units from all sessions.
"""

def run_one_neuron(session_id, neuron_idx, max_epochs, time_offset, device):
    try:
        json_path = f'dataset_single_unit/{session_id}/neuron_{neuron_idx}.json'
        if not os.path.exists(json_path):
            print(f"Neuron file not found: {json_path}")
            return

        data = pd.read_json(json_path)
        data['neuron_0'] = data['neuron_0'].apply(
            lambda x: np.array([x]) if isinstance(x, (int, float)) else np.array(x))
        data = data[data['neuron_0'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        if data.empty:
            print(f"Neuron {neuron_idx} skipped (no valid data)")
            return

        print(f"Running optimization for session {session_id}, neuron {neuron_idx}")
        single_unit_optimization(
            df=data,
            time_offset=time_offset,
            max_epochs=max_epochs,
            device=device,
            save_path='optimization_results_single_unit',
            session_id=session_id,
            neuron_id=neuron_idx
        )
    except Exception as e:
        print(f"Error on session {session_id}, neuron {neuron_idx}: {e}")

def get_sessions_and_neurons(dataset_dir='dataset_single_unit'):
    session_neuron_dict = {}
    for session_id in os.listdir(dataset_dir):
        session_path = os.path.join(dataset_dir, session_id)
        if not os.path.isdir(session_path):
            continue
        neuron_indices = []
        for fname in os.listdir(session_path):
            if fname.startswith('neuron_') and fname.endswith('.json'):
                try:
                    neuron_idx = int(fname.split('neuron_')[1].split('.json')[0])
                    neuron_indices.append(neuron_idx)
                except Exception:
                    continue
        if neuron_indices:
            session_neuron_dict[session_id] = sorted(neuron_indices)
    return session_neuron_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    args = parser.parse_args()

    data_path = 'dataset_single_unit'
    session_neuron_dict = get_sessions_and_neurons(data_path)

    for session_id, neuron_indices in session_neuron_dict.items():
        print(f"Processing session {session_id} with {len(neuron_indices)} neurons")
        Parallel(n_jobs=args.n_jobs)(
            delayed(run_one_neuron)(
                session_id, neuron_idx, args.max_epochs, 0.2, args.device
            ) for neuron_idx in tqdm(neuron_indices)
        )
        print(f"Finished processing session {session_id}")