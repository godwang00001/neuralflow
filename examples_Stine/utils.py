import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import neuralflow
from neuralflow.utilities.psth import extract_psth
from neuralflow.feature_complexity.fc_base import FC_tools
import pickle


def select_neuron(df: pd.DataFrame,
                 side_col: str = 'chosen_side',
                 stim_col: str = 'stim_onset',
                 rt_col:   str = 'RT',
                 spike_col: str = 'neuron_0',
                 left_label: str = 'left',
                 right_label: str = 'right',
                 coherence: float = 0.512):
    """
    Return True if the neuron passes all three criteria from the Nature paper.

    Required columns in `df`  (one trial per row):
        chosen_side, stim_onset, RT, neuron_0, Coherence, correct_response
    The entry in `neuron_0` must be an iterable of spike times (s starting at 0).
    """
    # # ---------------- Criterion 2: total # trials ----------------
    # if len(df) < 560:
    #     return False          # 直接失败，无须再算
    
    # Only keeps the data with the specified coherence
    df = df.copy()
    data = df[df['Coherence'] == coherence]

    # ---------------- 预先对齐两种参考系 ----------------
    # 刺激对齐
    data['spk_stim'] = data.apply(
        lambda x: x[spike_col] - x[stim_col], axis=1
    )
    # 眼跳(反应) 对齐
    data['spk_sacc'] = data.apply(
        lambda x: x[spike_col] - (x[stim_col] + x[rt_col]), axis=1
    )

    # ---------------- Criterion 1: PSTH ≥ 5 Hz ----------------
    fr_thres = 5.0
    med_rt   = data[rt_col].median()           # 秒
    bin_w    = 0.075                          # 75 ms
    bin_step = 0.010                          # 10 ms
    t_grid   = np.arange(0.0, med_rt-bin_w+1e-9, bin_step)

    passes_fr = False
    for side in (left_label, right_label):
        trials = data.loc[(data[side_col] == side) & (data['Coherence'] == coherence), 'spk_stim']
        if trials.empty:
            continue
        # 逐滑窗求 trial-avg firing-rate
        for t0 in t_grid:
            t1 = t0 + bin_w
            fr = np.mean([np.sum((spk >= t0) & (spk < t1)) for spk in trials]) / bin_w
            if fr >= fr_thres:
                passes_fr = True
                break
        if passes_fr:
            break
    if not passes_fr:
        return False

    # ---------------- Criterion 3: selectivity AUC > 0.6 --------
    def auc_in_window(series_spk, t0, t1):
        counts = series_spk.apply(lambda spk: np.sum((spk >= t0) & (spk < t1)))
        labels = (data[side_col] == right_label).astype(int)   # 1:right, 0:left
        if labels.nunique() < 2:                 # 只有一侧 → 无法算 AUC
            return 0.5
        auc = roc_auc_score(labels, counts)
        return max(auc, 1-auc)                  # 左高或右高都算 selectivity

    auc_after  = auc_in_window(data['spk_stim'], 0.0, 0.2)     # 0-0.2 s after stim
    auc_before = auc_in_window(data['spk_sacc'], -0.2, 0.0)    # –0.2-0 s before sacc
    if max(auc_after, auc_before) <= 0.6:
        return False

    # ---------------- All three criteria passed -----------------
    return True





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

def single_unit_optimization(df: pd.DataFrame,
                             time_offset: float = 0.1,
                             coherence_levels: list = [0.512, 0.256],
                             max_epochs: int = 50,
                             device: str = 'CPU',
                             save_path: str = 'optimization_results_single_unit',
                             session_id: str = '20201030',
                             neuron_id: int = 0):
    

    if device == 'GPU':
        with_cuda = True
    else:
        with_cuda = False

    

    ### 1. split the data into datasample 1 and datasample 2
    data = df.copy()
    datasample1, datasample2 = {}, {}
    for chosen_side in ['left', 'right']:
        for coherence in coherence_levels:

            # Filter data for the current condition
            data_cur = data[(data.chosen_side == chosen_side) & (data.Coherence == coherence)].reset_index()
            # Align to stimulus onset and subtract time offset
            data_cur = data_cur.assign(spikes = lambda x: x.neuron_0 - x.stim_onset - time_offset, axis = 1)
            
            # For fitting, only consider spikes after stimulus onset and before reaction time
            data_cur['spikes'] = data_cur.apply(lambda x: x.spikes[(x.spikes >= 0) & (x.spikes <= x.RT - time_offset)], axis = 1)

            # make sure the reaction time is greater than the time offset
            data_cur = data_cur[data_cur['RT'] >= time_offset].reset_index(drop=True) 

            # time epoch
            data_cur['time_epoch'] = data_cur.RT.apply(lambda x: (0, x - time_offset))
            
            # Let's put even trials into datasample 1, and odd trials into datasample 2
            num_trials = data_cur.shape[0]
            shuffled_trial_numbers = np.random.permutation(range(num_trials))
            
            ind1 = np.arange(0, num_trials, 2)
            datasample1[f'{chosen_side}_{coherence}'] = neuralflow.SpikeData(
                data = np.array([data_cur.loc[ind1,'spikes'],], dtype = object), dformat = 'spiketimes', time_epoch = data_cur.loc[ind1, 'time_epoch'].to_list(), with_cuda=with_cuda
            )
            ind2 = np.arange(1, num_trials, 2)
            datasample2[f'{chosen_side}_{coherence}'] = neuralflow.SpikeData(
                data = np.array([data_cur.loc[ind2,'spikes'],], dtype = object), dformat = 'spiketimes', time_epoch = data_cur.loc[ind2, 'time_epoch'].to_list(), with_cuda=with_cuda
            )
            
            # Convert to ISI format
            datasample1[f'{chosen_side}_{coherence}'].change_format('ISIs')
            datasample2[f'{chosen_side}_{coherence}'].change_format('ISIs')


    
    ### 2. optimize the model
    # The optimization was performed in a grid with Np = 8, Ne = 64. Here we set Ne to 16 to reduce fitting time
    grid = neuralflow.GLLgrid(Np = 8, Ne = 16, with_cuda=with_cuda)

    # Initial guess
    init_model = neuralflow.model.new_model(
        peq_model = {"model": "uniform", "params": {}},
        p0_model = {"model": "cos_square", "params": {}},
        D = 1,
        fr_model = [{"model": "linear", "params": {"slope": 1, "bias": 100}}],
        params_size={'peq': 4, 'D': 1, 'fr': 1, 'p0': 1},
        grid = grid,
        with_cuda=with_cuda
    )

    optimizer = 'ADAM'
    # The default max_epochs is 50 for a quick optimization. However, to get the best model it usually requires ~300 epochs
    opt_params = {'max_epochs': max_epochs, 'mini_batch_number': 20, 'params_to_opt': ['F', 'F0', 'D', 'Fr', 'C'], 'learning_rate': {'alpha': 0.05}}
    ls_options = {'C_opt': {'epoch_schedule': [0,1,5,30], 'nSearchPerEpoch': 3, 'max_fun_eval': 2}, 'D_opt': {'epoch_schedule': [0,1,5,30], 'nSearchPerEpoch': 3, 'max_fun_eval': 25}}
    boundary_mode = 'absorbing'

    # Train on datasample 1
    dataTR = [v for v in datasample1.values()]
    optimization1 = neuralflow.optimization.Optimization(
                        dataTR,
                        init_model,
                        optimizer,
                        opt_params,
                        ls_options,
                        boundary_mode=boundary_mode,
                        device=device
                    )

    # run optimization
    print('Running optimization on datasample 1')
    optimization1.run_optimization()

    # Train on datasample 2
    dataTR = [v for v in datasample2.values()]
    optimization2 = neuralflow.optimization.Optimization(
                        dataTR,
                        init_model,
                        optimizer,
                        opt_params,
                        ls_options,
                        boundary_mode=boundary_mode,
                        device=device
                    )
    
    # run optimization
    print('Running optimization on datasample 2')
    optimization2.run_optimization()


    ### Compute the feature consistency and reflect the model if needed
    JS_thres = 0.0015
    FC_stride = 5
    smoothing_kernel = 10

    fc = FC_tools(non_equilibrium=True, model=init_model, boundary_mode=boundary_mode, terminal_time=1)
    FCs1, min_inds_1, FCs2, min_inds_2, JS, FC_opt_ind = (
        fc.FeatureConsistencyAnalysis(optimization1.results, optimization2.results, JS_thres, FC_stride, smoothing_kernel)
        )

    invert = fc.NeedToReflect(optimization1.results, optimization2.results)

    ### save the results
    n_conds = len(coherence_levels) * 2
    opt_ind_1 = min_inds_1[FC_opt_ind]
    opt_ind_2 = min_inds_2[FC_opt_ind]
    Phi_1 = {cond: -np.log(optimization1.results['peq'][opt_ind_1][cond])*optimization1.results['D'][opt_ind_1][0] for cond in range(n_conds)}
    Phi_2 = {cond: -np.log(optimization2.results['peq'][opt_ind_2][cond])*optimization2.results['D'][opt_ind_2][0] for cond in range(n_conds)}
    p0_1 = optimization1.results['p0'][opt_ind_1][0]
    p0_2 = optimization2.results['p0'][opt_ind_2][0]
    fr_1 = optimization1.results['fr'][opt_ind_1][0,...,0]
    fr_2 = optimization2.results['fr'][opt_ind_2][0,...,0]
    D_1 = optimization1.results['D'][opt_ind_1][0]
    D_2 = optimization2.results['D'][opt_ind_2][0]
    result = {
        'session_id': session_id,
        'neuron_id': neuron_id,
        'FCs1': FCs1,
        'min_inds_1': min_inds_1,
        'FCs2': FCs2,
        'min_inds_2': min_inds_2,
        'JS': JS,
        'FC_opt_ind': FC_opt_ind,
        'invert': invert,
        'x_d': init_model.grid.x_d,
        'Phi_1': Phi_1,
        'Phi_2': Phi_2,
        'p0_1': p0_1,
        'p0_2': p0_2,
        'fr_1': fr_1,
        'fr_2': fr_2,
        'D_1': D_1,
        'D_2': D_2,
    }

    with open(f'{save_path}/{session_id}_neuron_{neuron_id}.pkl', 'wb') as f:
        pickle.dump(result, f)




def plot_optimization_results(result_path: str = 'optimization_results_single_unit',
                              save_path: str = 'figs/',
                              session_id: str = '20201030',
                             neuron_id: int = 0):
    
    ### load the optimization results
    with open(f'{result_path}/{session_id}_neuron_{neuron_id}.pkl', 'rb') as f:
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

    plt.tight_layout()
    plt.savefig(f'{save_path}/{session_id}_neuron_{neuron_id}.png')






    