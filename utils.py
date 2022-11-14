import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import copy

# only for comparison with CREPE
def visualize(list_of_dicts_time_frequency_dfs, labeling_hz, t=5, t_scale=1, f_scale=1):
    '''
    Input: [  {'estimator name' (or 'ground'): (time, frequency) dataframe}
              {other track prediction dicts in the same fashion}...,
              or,
              activation matrix (ndarray) of a neural network output ]
    '''
    dur_s = t
    colors = ['r', 'k', 'b', 'g', 'm', 'darkorange', 'y']
    yticks = [196, 293.7, 440, 660, 987.77]
    
    yticklabels = [r'$\mathrm{G_{3}}$', r'$\mathrm{D_{4}}$', r'$\mathrm{A_{4}}$', r'$\mathrm{E_{5}}$', r'$\mathrm{B_{5}}$']
    min_label = np.argmax(labeling_hz[labeling_hz<yticks[0]-20])
    fmin, fmax = labeling_hz[min_label], labeling_hz[-1]
    ylim = [fmin, fmax]
    cm = 2.2 / 2.54  # centimeters in inches'$\mathrm{D_{4}}$'
    fig, axes = plt.subplots(len(list_of_dicts_time_frequency_dfs), 1, sharex=True, figsize=((2*dur_s*cm*f_scale), (9/4)*len(list_of_dicts_time_frequency_dfs)*cm*f_scale))
    try:
      xlim = max([next(iter(dict_of_time_frequency_dfs.values()))['time'].iloc[-1] for dict_of_time_frequency_dfs in list_of_dicts_time_frequency_dfs])
    except AttributeError:
      xlim = next(iter(list_of_dicts_time_frequency_dfs[0].values()))['time'].iloc[-1]
    if not hasattr(axes, '__len__'):
      axes = [axes]
    for i, dict_of_time_frequency_dfs in enumerate(list_of_dicts_time_frequency_dfs):
      try:
        dur = next(iter(dict_of_time_frequency_dfs.values()))['time'].iloc[-1]
        for j, (label, time_frequency_df) in enumerate(dict_of_time_frequency_dfs.items()):
          time_frequency_df.plot.scatter(x='time', y='frequency', ax = axes[i], grid=True, xlim=[0, xlim], ylim=ylim, yticks=None, logy=True, label=label, color=colors[j], alpha=0.7, s=3)
        axes[i].set_yticks(yticks)
      except: #then it is activation of a nn
        act = dict_of_time_frequency_dfs[:, min_label:].T
        cmap = {'r':'Reds', 'k':'Greys', 'b':'Blues', 'g':'Greens', 'm':'Purples', 'darkorange':'Oranges', 'y':'YlOrBr'}
        axes[i].imshow(act, aspect="auto", origin="lower", cmap=cmap[colors[i]], extent=[0, xlim, np.log(fmin), np.log(fmax)])
        axes[i].grid()
        axes[i].set_yticks(np.log(yticks))
        axes[i].set_ylabel('activations')
      
      axes[i].set_yticklabels([],minor=True)
      axes[i].set_yticklabels(yticklabels, fontsize=14)
    xticks = range(0, int(dur)+1, int(1))
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels(list(range(len(xticks))), fontsize=14)
    axes[i].set_xlabel('time in seconds')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.close()
    return fig


# To make sonification more appealing (Just a gimmick for now. Raw confidence thresholding in TAPE does not get on well with the sonification function)
def silence_unvoiced_segments(pitch_track_csv, low_confidence_threshold=0.2,
                              high_confidence_threshold=0.7, min_voiced_segment_ms=12):
  
    def silence_segments_one_run(confidences, confidence_threshold, segment_len_th):
        conf_bool = np.array(confidences > confidence_threshold).reshape(-1)
        absdiff = np.abs(np.diff(np.concatenate(([False], conf_bool, [False]))))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        segment_durs = np.diff(ranges, axis=1)
        valid_segments = ranges[np.repeat(segment_durs > segment_len_th, repeats=2, axis=1)].reshape(-1, 2)
        voiced = np.zeros(len(confidences), dtype=bool)
        for segment in valid_segments:
            voiced[segment[0]:segment[1]] = True
        return voiced

    annotation_interval_ms = int(round(1000 * pitch_track_csv.loc[:1, "time"].diff()[1]))
    voiced_th = int(np.ceil(min_voiced_segment_ms / annotation_interval_ms))

    # we do not accept the segment if a close neighbors do not have a confidence > 0.7
    smoothened_confidences = medfilt(pitch_track_csv["confidence"], kernel_size=2 * (voiced_th // 2) + 1)
    smooth_voiced = silence_segments_one_run(smoothened_confidences,
                                             confidence_threshold=high_confidence_threshold, segment_len_th=voiced_th)

    # we also do not accept the pitch values if the individual confidences are really low
    hard_voiced = silence_segments_one_run(pitch_track_csv["confidence"],
                                           confidence_threshold=low_confidence_threshold, segment_len_th=voiced_th)

    # we accept the intersection of these two zones
    voiced = np.logical_and(smooth_voiced, hard_voiced)

    smoothened_pitch = copy.deepcopy(pitch_track_csv["frequency"])
    smoothened_pitch[~voiced] = np.nan
    smoothened_pitch.fillna(smoothened_pitch.rolling(window=voiced_th, 
                                                     min_periods=1+voiced_th//2).median(), inplace=True)

    absdiff = np.abs(np.diff(np.concatenate(([False], voiced, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    unvoiced_ranges = np.vstack([ranges[:-1, 1], ranges[1:, 0]]).T
    for unvoiced_boundary in unvoiced_ranges:
        # we don't want small unvoiced zones. Check if they are acceptable with a more favorable mean thresholding
        len_unvoiced = np.diff(unvoiced_boundary)[0]
        if len_unvoiced < 2*voiced_th:
            avg_confidence = pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "confidence"].mean()
            if 2*avg_confidence > (low_confidence_threshold+high_confidence_threshold):
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True
            elif len_unvoiced < voiced_th:
                pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "frequency"] = \
                    smoothened_pitch[unvoiced_boundary[0]:unvoiced_boundary[1]]
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True

    pitch_track_csv.loc[~voiced, "frequency"] = 0
    pitch_track_csv["frequency"] = pitch_track_csv["frequency"].fillna(0)
    return pitch_track_csv
