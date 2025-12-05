import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from neuralflow.utilities.psth import extract_psth

def plot_psth(df: pd.DataFrame,
              Coherence: float = 0.512,
              neuron_id: str = 'neuron_0',
              session_id: str = '20201030'):
    data_spiketimes_stim, data_spiketimes_sacc = {}, {}
    RTs = {}

    coherence_levels = [0.512, 0.256]

    for chosen_side in ['left', 'right']:
        for coherence in coherence_levels:
            # Filter data for the current condition
            data_cur = df[(df.chosen_side == chosen_side) & (df.Coherence == coherence)]
            
            # Align to stimulus onset
            data_cur = data_cur.assign(spikes_stim = lambda x: x.neuron_0 - x.stim_onset, axis = 1)
            # Align to saccade onset (stim_onset + RT)
            data_cur = data_cur.assign(spikes_sacc = lambda x: x.neuron_0 - (x.stim_onset + x.RT), axis = 1)
            
            # For visualization, consider time window from stimulus_onset-0.1 to RT 
            data_cur['spikes_stim'] = data_cur.apply(lambda x: x.spikes_stim[(x.spikes_stim >= -0.1) & (x.spikes_stim <= x.RT)], axis = 1)
            # For saccade aligned, consider -0.5s to +0.1s around saccade
            data_cur['spikes_sacc'] = data_cur.apply(lambda x: x.spikes_sacc[(x.spikes_sacc >= -0.5) & (x.spikes_sacc <= 0.1)], axis = 1)
            
            # Record reaction time and spikes
            RTs[f'{chosen_side}_{coherence}'] = data_cur['RT'].to_list()
            data_spiketimes_stim[f'{chosen_side}_{coherence}'] = np.array([data_cur['spikes_stim'].to_list(),], dtype = object)
            data_spiketimes_sacc[f'{chosen_side}_{coherence}'] = np.array([data_cur['spikes_sacc'].to_list(),], dtype = object)
            




    # Plot parameters
    time_window = 0.075 # bin size
    dt = 0.01 # bin step
    color_lines = {
        'left_0.512': '#FF0000',      # Bright red
        'left_0.256': '#FF8080',      # Light red
        'right_0.512': '#00FF00',     # Bright green  
        'right_0.256': '#80FF80'      # Light green
    }

    # Create 1x2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot stimulus aligned PSTH
    tbegin_stim = -0.1 # start time for stim PSTH
    tend_stim = 0.500 # End time for stim PSTH

    for chosen_side in ['left', 'right']:
        for coherence in coherence_levels:
            tb, rates, rates_SEM = extract_psth(data_spiketimes_stim[f'{chosen_side}_{coherence}'], 
                                            RTs[f'{chosen_side}_{coherence}'], 
                                            time_window, dt, tbegin_stim, tend_stim) 
            spikes = rates[0]
            time = tb[0][:spikes.size]
            err = rates_SEM[0]
            ax1.plot(time, spikes, '-', color=color_lines[f'{chosen_side}_{coherence}'], label=f'{chosen_side}_{coherence}')
            ax1.fill_between(time, spikes-err, spikes+err, 
                            color=color_lines[f'{chosen_side}_{coherence}'],
                            alpha=0.5, edgecolor="none")
    ax1.set_xlabel('Time aligned to stimulus onset, s')
    ax1.legend()
    ax1.set_ylabel('PSTH, Hz')

    # Plot saccade aligned PSTH
    tbegin_sacc = -0.5 # start time for saccade PSTH
    tend_sacc = 0.1 # End time for saccade PSTH

    for chosen_side in ['left', 'right']:
        for coherence in coherence_levels:
            tb, rates, rates_SEM = extract_psth(data_spiketimes_sacc[f'{chosen_side}_{coherence}'], 
                                            RTs[f'{chosen_side}_{coherence}'], 
                                            time_window, dt, tbegin_sacc, tend_sacc) 
            spikes = rates[0]
            time = tb[0][:spikes.size]
            err = rates_SEM[0]
            ax2.plot(time, spikes, '-', color=color_lines[f'{chosen_side}_{coherence}'])
            ax2.fill_between(time, spikes-err, spikes+err, 
                            color=color_lines[f'{chosen_side}_{coherence}'],
                            alpha=0.5, edgecolor="none")
    ax2.set_xlabel('Time aligned to saccade, s')
    fig.suptitle(session_id + ' ' + neuron_id)
    plt.tight_layout()


def plot_optimization_results(result_path: str = 'optimization_results_single_unit',
                              save_path: str = 'figs/',
                             neuron_id: int = 0,
                             if_save: bool = True):
    
    ### load the optimization results
    with open(f'{result_path}/neuron_{neuron_id}.pkl', 'rb') as f:
        res = pickle.load(f)



    ### plot the optimization results
    JS_thres = 0.0015
    color_lines = ['#FF8C8C', [0.431, 0.796, 0.388], [1, 0.149, 0], [0, 0.561, 0]]

    fig = plt.figure(figsize = (20, 10))
    gs = gridspec.GridSpec(2,4)

    # Plot JS diveregence vs. FC
    ax = plt.subplot(gs[0, 0])
    ax.plot(res['FCs1'], res['JS'], linewidth = 3, color = [120/255, 88/255, 170/255], label = 'JS curve')
    ax.plot(res['FCs1'][res['FC_opt_ind']], res['JS'][res['FC_opt_ind']], '.', markersize=16, color = '#87A2FB', label = 'Selected FC')
    ax.plot(res['FCs1'], JS_thres*np.ones_like(res['FCs1']), '--', linewidth = 1, color = [0.4]*3, label = 'JS threshold')
    plt.legend()
    plt.xlabel('Feature complexity')
    plt.ylabel('Jason-Shanon divergence')

    # Plot negative relative scaled to start at 0)
    # Note that for various reasons logliks may not monotinically decrease for the first few iteration.


    opt_ind_1 = res['min_inds_1'][res['FC_opt_ind']]
    opt_ind_2 = res['min_inds_2'][res['FC_opt_ind']]
    for cond in range(4):
        ax = plt.subplot(gs[cond//2, cond%2+1])
        # Note that original potential needs to be scaled back by D because we use non-conventional form of Langevin and Fokker-Planck equation.
        Phi1 = res['Phi_1'][cond]
        Phi2 = res['Phi_2'][cond]
        plt.plot(res['x_d'], Phi1, linewidth = 2, color = color_lines[cond])
        plt.plot(res['x_d'], Phi2[::-1 if res['invert'] else 1], linewidth = 2, color = color_lines[cond])
        plt.xlabel('Latent state, $x$')
        plt.ylabel('Potential $\Phi(x)$')
        
    ax = plt.subplot(gs[0,3])
    plt.plot(res['x_d'], res['p0_1'], linewidth = 2, color = 'black')
    plt.plot(res['x_d'], res['p0_2'][::-1 if res['invert'] else 1], linewidth = 2, color = 'black')
    plt.ylabel('$p_0(x)$')
    plt.xlabel('Latent state, $x$')


    fr_color = [28/255, 117/255, 188/255]
    ax = plt.subplot(gs[1,3])
    fr1 = res['fr_1']
    fr2 = res['fr_2']
    plt.plot(res['x_d'], fr1, linewidth = 2, color = fr_color)
    plt.plot(res['x_d'], fr2[::-1 if res['invert'] else 1], linewidth = 2, color = fr_color)
    plt.xlabel('Latent state, $x$')
    plt.ylabel('Firing rate $f(x)$, Hz')
    plt.title(f'Neuron {neuron_id}')

    plt.tight_layout()
    if if_save:
        plt.savefig(f'{save_path}/neuron_{neuron_id}.png')
    return fig


def plot_heatmap_tuning_curve(fr_1_all, fr_2_all, save_path: str = 'figs/', if_save: bool = True):

    """
    Plot the heatmap of the tuning curves for a population.
    Tuning curves are fitted over the first half of the trials and the second half of the trials, respectively.
    In the first row, the neurons are sorted by the peak time of the tuning curve of the first half of the trials,
    and in the second row, the neurons are sorted by the peak time of the tuning curve of the second half of the trials.

    Args:
        fr_1_all: numpy array of shape (num_neurons, num_x)
        fr_2_all: numpy array of shape (num_neurons, num_x)
        save_path: str, path to save the figure
    """
    assert fr_1_all.shape == fr_2_all.shape, 'fr_1_all and fr_2_all must have the same shape.'

    num_neurons, num_x = fr_1_all.shape
    fr_1_all_norm = (fr_1_all - np.min(fr_1_all, axis=1)[:,np.newaxis]) / (np.max(fr_1_all, axis=1) - np.min(fr_1_all, axis=1))[:,np.newaxis]
    fr_2_all_norm = (fr_2_all - np.min(fr_2_all, axis=1)[:,np.newaxis]) / (np.max(fr_2_all, axis=1) - np.min(fr_2_all, axis=1))[:,np.newaxis]

    # Get peak times for each neuron
    peak_times_1 = np.argmax(fr_1_all_norm, axis=1)
    peak_times_2 = np.argmax(fr_2_all_norm, axis=1)

    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2)

    # Sort based on peak times
    sort_idx_1 = np.argsort(peak_times_1)
    sort_idx_2 = np.argsort(peak_times_2)

    # Use same color limits for both plots
    vmin = 0
    vmax = 1

    # Common axis settings
    x_ticks = [0, num_x//2, num_x-1]
    x_labels = [-1, 0, 1]
    y_ticks = [0, num_neurons-1]
    y_labels = [1, num_neurons]

    titles = ['First half sorted on first half',
             'Second Half sorted on first half',
             'First half sorted on second half',
             'Second Half sorted on second half']
    
    data = [fr_1_all_norm[sort_idx_1], fr_2_all_norm[sort_idx_1],
            fr_1_all_norm[sort_idx_2], fr_2_all_norm[sort_idx_2]]

    for i in range(4):
        ax = plt.subplot(gs[i])
        im = ax.imshow(data[i], aspect='auto', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(x_ticks, x_labels)
        ax.set_yticks(y_ticks, y_labels)
        ax.tick_params(labelsize=15)
        
        ax.set_xlabel(r'Latent state, $x$', fontsize=18)
        if i % 2 == 0:  # Left column
            ax.set_ylabel('# Neuron', fontsize=18)
        ax.set_title(titles[i], fontsize=24)

    plt.tight_layout()
    # # Add colorbar in the dedicated space
    # cbar = plt.colorbar(im1, cax=cax)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('Normalized Tuning curve', fontsize=18)
    # # Set colorbar ticks to only show 0 and 1
    # cbar.set_ticks([0, 1])
    # # Make colorbar thinner
    # cax.set_box_aspect(5)  # Increase this number to make the colorbar thinner

    


    