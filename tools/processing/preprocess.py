import argparse
import math
import os
import pandas as pd
import time
import warnings
import sklearn
import numpy as np

from tqdm import tqdm
from tqdm.auto import trange

warnings.filterwarnings("ignore", category=FutureWarning)  # frame.append is deprecated - pandas.concat instead


# --loadpath
# kirun-speed-estimation\data\metadata\original
# --savepath
# kirun-speed-estimation\data\metadata\wdw5
# --window-size
# 5.0 (regression) 200 (sed)

def parse_args():
    """Returns: cl args"""
    parser = argparse.ArgumentParser('KIRun Speed Training')
    parser.add_argument('--loadpath', help='data/metadata/original', required=True)
    parser.add_argument('--savepath', help='data/metadata/<savepath_name>', required=True)
    parser.add_argument('--window-size', type=float, default=5.0)
    args = parser.parse_args()
    return args


# ###################
# APPLY WINDOWING::
# ###################

def create_windowed_folds(_loadpath, _savepath, window_function, window_size=5.0, folds=[0, 1, 2, 3, 4]):
    """
    Applies windowing (step/secs) to fold_n (containing train,dev,test).
    Converts and appends the results into .csv and writes the .csv to the savepath.
    Args:
        _loadpath:
        _savepath: should lead to a folder which is designated for the given window_size
        window_function:
        window_size: amount of seconds in which stepCounts are summed up for
        folds: preexisting train,val,test splits in kirun dataset, if not available use create_windowed_split
    """
    os.makedirs(_savepath, exist_ok=True)
    t0 = time.time()

    print("Start windowing.")
    for k in folds:
        loadpath = os.path.join(_loadpath, f'fold{k}/')
        os.makedirs(_savepath + f'/fold{k}', exist_ok=True)
        savepath_current = _savepath + f'/fold{k}'

        # window train, dev, test
        d_train = pd.read_csv(loadpath + 'steps_train.csv')
        d_train_windowed = window_function(d_train, window_size)
        print(f"Fold {k} train windowed.", time.time() - t0, "s")
        os.makedirs(loadpath, exist_ok=True)
        d_train_windowed.to_csv(savepath_current + f'/train.csv')

        d_dev = pd.read_csv(loadpath + 'steps_dev.csv')
        d_dev_windowed = window_function(d_dev, window_size)
        print(f"Fold {k} dev windowed.", time.time() - t0, "s")
        os.makedirs(loadpath, exist_ok=True)
        d_dev_windowed.to_csv(savepath_current + f'/devel.csv')

        d_test = pd.read_csv(loadpath + 'steps_test.csv')
        d_test_windowed = window_function(d_test, window_size)
        print(f"Fold {k} test windowed.", time.time() - t0, "s")
        os.makedirs(loadpath, exist_ok=True)
        d_test_windowed.to_csv(savepath_current + f'/test.csv')

        # combine into fold_k.csv
        fold = pd.concat([d_train_windowed, d_dev_windowed, d_test_windowed])
        os.makedirs(savepath_current, exist_ok=True)
        fold.to_csv(savepath_current + f'/fold{k}.csv')
        print(f"Fold {k} windowed.")

    print("Windowing finished.", time.time() - t0, "s")


def create_new_windowed_split(_loadpath, _savepath, window_function, window_size=5.0, sessioncount=1.0):
    """
    Applies windowing (step/secs) to steps.csv.
    Creates new train, val, test split based on total amount of samples.
    Split: 80/10/10 (can be adapted to different split size, original metadata contains ~ 60/20/20 splits)
    TODO: this split is not speaker independt
    Args:
        _loadpath: should lead to a folder with all folds and steps.csv files (=./metadata/ initially)
        _savepath: should lead to a folder which is designated for the given window_size
        window_function:
        window_size: amount of seconds in which stepCounts are summed up for
        sessioncount: factor on how many running sessions should be contained
    """
    os.makedirs(_savepath, exist_ok=True)

    t0 = time.time()
    print("Start windowing.")

    loadpath = os.path.join(_loadpath, f'steps.csv')
    os.makedirs(_savepath + f'/combined', exist_ok=True)
    savepath_current = _savepath + f'/combined'

    # Read steps in DataFrame
    data = pd.read_csv(loadpath)

    # get unique runs:
    runs = data['session'].unique()  # 196

    # shuffle ordering
    shuffled_runs = sklearn.utils.shuffle(runs)

    # determine split size
    count = int(len(shuffled_runs) * sessioncount)
    shuffled_runs = shuffled_runs[:count]
    length = len(shuffled_runs)

    # create run-based split 60/20/20
    train_runs = shuffled_runs[:int(length * 0.6)]
    val_runs = shuffled_runs[int(length * 0.6):int(length * 0.8)]
    test_runs = shuffled_runs[int(length * 0.8):]

    # Split data into train, validation, and test sets in 80/10/10 based on shuffle:
    train_data = data[data['session'].str.contains(train_runs[0])]
    for i in range(1, len(train_runs)):
        train_data = train_data.append(data[data['session'].str.contains(train_runs[i])], ignore_index=True)

    val_data = data[data['session'].str.contains(val_runs[0])]
    for i in range(1, len(val_runs)):
        val_data = val_data.append(data[data['session'].str.contains(val_runs[i])], ignore_index=True)

    test_data = data[data['session'].str.contains(test_runs[0])]
    for i in range(1, len(test_runs)):
        test_data = test_data.append(data[data['session'].str.contains(test_runs[i])], ignore_index=True)

    # window train, dev, test splits:
    d_train_windowed = window_function(train_data, window_size)
    d_train_windowed.to_csv(savepath_current + f'/train.csv')

    d_dev_windowed = window_function(val_data, window_size)
    d_dev_windowed.to_csv(savepath_current + f'/devel.csv')

    d_test_windowed = window_function(test_data, window_size)
    d_test_windowed.to_csv(savepath_current + f'/test.csv')

    # combine into one .csv with windowed portions (needed for melspects.py)
    split = pd.concat([d_train_windowed, d_dev_windowed, d_test_windowed])
    os.makedirs(savepath_current, exist_ok=True)
    split.to_csv(savepath_current + f'/split.csv')

    print("Windowing finished.", time.time() - t0, "s")


# ###################
# REGRESSION::
# ###################

def regression_window(data, window_size, both=True, right=True):
    """
    This was used for all experiments on regression.

    Windows audio based on a fixed amount of time i.e. 5, 10 or 20 seconds.
    Assumes that each row corresponds to one step (left or right are counted separately)
    Windows are closed at the last occuring step within the window_size, e.g. [0.0-4.79,4.79-9.69,...].

    Args:
        data: read-in-.csv file containing the information and audio-reference to runs
        window_size: amount of seconds in which steps are being counted
        both: False counts only one footstep depending on parameter right
        right: True counts right, False counts left footsteps
    Returns:
        DataFrame with 4 columns: file,start,end,steps
    """

    columns = ['file', 'start', 'end', 'steps']
    d_new = pd.DataFrame(columns=columns)
    counter = 0
    left_counter = 0
    right_counter = 0
    frame_start = 0.0

    for i, row in data.iterrows():
        if i == 0 or row['elapsed'] == 0.0:  # new run
            frame_start = 0.0  # records the elapsed time at the start of each window
            if row['foot'] == 'right':
                right_counter += 1
            if row['foot'] == 'left':
                left_counter += 1
            counter += 1
            continue
        else:  # next step
            if row['foot'] == 'right':
                right_counter += 1
            if row['foot'] == 'left':
                left_counter += 1
            counter += 1

        # add entry or abort within window_size
        # monitor (i+1)th timestep to abort within the given window_size (last window can be shorter)
        if (i + 1 == len(data)) or data.iloc[i + 1].elapsed - frame_start >= window_size:
        #if i == len(data) or data.iloc[i].elapsed - frame_start >= window_size:
            if both:
                entry = {'file': row['file'], 'start': frame_start, 'end': row['elapsed'], 'steps': counter}
                d_new = d_new.append(entry, ignore_index=True)
            else:
                if right:
                    entry = {'file': row['file'], 'start': frame_start, 'end': row['elapsed'], 'steps': left_counter}
                    d_new = d_new.append(entry, ignore_index=True)
                else:
                    entry = {'file': row['file'], 'start': frame_start, 'end': row['elapsed'], 'steps': right_counter}
                    d_new = d_new.append(entry, ignore_index=True)

            frame_start = row['elapsed']  # update to new
            counter = 0
            left_counter = 0
            right_counter = 0

    return d_new


def segmented_regression_window(data, window_size):
    """
    Windows audio based on a fixed amount of time i.e. 5, 10 or 20 seconds.
    Assumes that each row corresponds to one step (left or right are counted separately)

    Windows are closed after the last occuring step at window_size threshold, e.g. [0.0-5.0, 5.0-10.0,...].
    Results are worse than with regression_window after training on selected networks.

    Args:
        data: read-in-.csv file containing the information and audio-reference to runs
        window_size: amount of seconds in which steps are being counted
    Returns:
        DataFrame with 4 columns: file,start,end,steps
    """
    assert window_size in [5, 10, 20], 'ideal window_sizes are 5, 10, 20 [s]'

    columns = ['file', 'start', 'end', 'steps']
    d_new = pd.DataFrame(columns=columns)

    counter = 0
    segment_start = 0.0  # records the segment theshold at the start of each window
    segment_end = segment_start + window_size  # records the segment theshold at the end of each window

    for i, row in data.iterrows():
        if i == 0 or row['elapsed'] == 0.0:  # new run
            segment_start = 0.0
            segment_end = segment_start + window_size  # 5.0 in first iter
            counter = 1  # first step of new run
            continue
        else:
            counter += 1  # next step

        # add entry or abort within window_size (when run is over)
        if (i + 1 == len(data)) or data.iloc[i + 1].elapsed - segment_start >= window_size:
            entry = {'file': row['file'], 'start': segment_start, 'end': segment_end, 'steps': counter}
            d_new = d_new.append(entry, ignore_index=True)
            # update counter and indexing vars for next window:
            segment_start = segment_end
            segment_end = segment_start + window_size
            counter = 0

    return d_new


# ###################
# SOUND EVENT DETECTION::
# ###################


def sed_window(data, window_size=200):
    """
        Inspired by https://arxiv.org/abs/2107.05463

        Form: [file, start, end, True/False]

        Windowing based on presence and absence of steps in a small time frame. Our annotations didn't include start and
        end time of step occurances, therefore we had to manually engineer them.
        Windowing based on partitioning run-sessions into segments of size <window_size>.
        -> Activity/Step occurances are assigned to each segment.
        Windowing takes a long time if all sessions are to be windowed. Possibility to window with sessioncount variable
        in create_new_windowed_split.

    Args:
        data: pandas dataframe containing steps.csv rows
        window_size: amount of milliseconds [ms] in which to monitor activeness of steps
    Returns: windowed dataframe

    """

    assert 10 <= window_size <= 200, 'window_size should be between 10 and 200 [ms], to represent a step.'
    columns = ['file', 'start', 'end', 'active']  # active is a binary label to mark the presence of a step in a window
    d_new = pd.DataFrame(columns=columns)  # windowed entries go here
    running_sessions = data['file'].unique()  # get all running session of the current fold:

    # iterate over all running sessions:
    for session in tqdm(running_sessions, desc='Session', total=len(running_sessions)):
        total_elapsed = max(data[data['file'].str.contains(session)].elapsed)
        total_segments = math.ceil(total_elapsed * 1000 / window_size)
        session_steps = list(data[data['file'].str.contains(session)].elapsed)  # seconds

        segment_start = 0.0  # first recorded step always start at 0.0
        segment_end = segment_start + round(window_size / 1000, 1)

        for i in range(total_segments):
            t_occ = [i for i in session_steps if segment_start <= i <= segment_end]  # list for noise (multiple)
            active = True if len(t_occ) != 0 else False
            entry = {'file': session, 'start': round(segment_start, 1), 'end': round(segment_end, 1), 'active': active}
            d_new = d_new.append(entry, ignore_index=True)
            segment_start = segment_end
            segment_end = segment_start + round(window_size / 1000, 1)

    return d_new


def sed_segmented_window(data, window_size=200, total_size=1.0):
    """
        Inspired by https://arxiv.org/abs/2107.05463

        Form: [file, start, end, '1.0, 0.0, 0.0, ..., 1.0']

        Difference to sed_window: activity annotations are 'filled up' in one column until a larger window_size is filled.
        Example: window_size 5s, segment_size 200ms => [file, start, end, [True, b_2, b_3 b_4, b_5, ..., b_25]].
        This way the training loader refers to multiple small windows but with bigger context.

    Args:
        data: pandas dataframe containing steps.csv rows
        window_size:  amount of milliseconds [ms] for a single segment
        total_size:  amount of seconds [s] in which to close the window of segments
    Returns: windowed dataframe

    """

    num_window_segments = (total_size * 1000) // window_size  # default=25 ;
    print('num_window_segments:', num_window_segments)

    columns = ['file', 'start', 'end', 'activities']
    # for i in range(int(num_window_segments)):  # multiple activities
    #     columns.append(f'a{i}')

    d_new = pd.DataFrame(columns=columns)  # windowed entries go here
    running_sessions = data['file'].unique()  # get all running session of the current fold:

    first_run = True

    # iterate over all running sessions:
    for session in tqdm(running_sessions, desc='Session', total=len(running_sessions)):

        # acquire the total elapsed time [seconds] for the current session:
        total_elapsed = max(data[data['file'].str.contains(session)].elapsed)
        if first_run:
            print('\n', total_elapsed)  # - 1247.482

        # calculate amount of segments:
        total_segments = math.ceil(total_elapsed * 1000 / window_size)
        if first_run:
            print(total_segments)  # - 6238 - [0.0, 0.2, 0.4, 0.6, 0.8, 1[s], ...

        # store step timestamps of this session:
        session_steps = list(data[data['file'].str.contains(session)].elapsed)  # seconds, in order
        if first_run:
            print(session_steps[
                  :20])  # - [0.0, 0.638, 1.989, 2.518, 3.04, 3.54, 3.94, 4.349, 4.749, 5.129, 5.528, 5.908, ...
        # session_steps = [int(s * 1000) for s in session_steps]  # milliseconds

        # now iterate over window, monitor step-occurances and label activeness:
        window_start = 0.0
        window_end = window_start + total_size

        segment_start = 0.000  # first recorded step always start at 0.0
        segment_end = segment_start + round(window_size / 1000, 4)  # adapt precision here
        if first_run:
            print('segment_start', segment_start, 'segment_end', segment_end)  # segment_start 0.0 segment_end 0.2

        window_activities = []
        first_run = False

        for i in range(total_segments):
            t_occ = [i for i in session_steps if segment_start <= i <= segment_end]  # list for noise (multiple)
            active = True if len(t_occ) != 0 else False
            window_activities.append(active)

            # here instead of adding segmentwise, we add after current window is 'full'
            if len(window_activities) == num_window_segments:
                window_activities = [int(b) for b in window_activities]  # bool -> numerical
                entry = {'file': session, 'start': round(window_start, 4), 'end': round(window_end, 4),
                         'activities': window_activities}

                d_new = d_new.append(entry, ignore_index=True)
                assert d_new.shape[1] == (4), d_new.shape[1]  # check correct number of columns (redundant)
                window_activities = []
                window_start += total_size
                window_end = window_start + total_size

            # still track the 'old', individual segments on their own
            segment_start = segment_end
            segment_end = segment_start + round(window_size / 1000, 4)

    return d_new


def sed_threshold_segmented_window(data, window_size=200, total_size=5.0, threshold=0.4):
    """
        Segment-based SED with thresholds
        Difference to sed_window_concatenated: True label based on thresholds of wave-proportion contained in segment
    TODO: not fully implemented
    Examples:
        segment s = [start, end] = [0.4, 0.6]
        step happens at 0.6, step duration ~ 200ms, step happens during d = [0.5, 0.7]
            if threshold is 50%:
            -> step is in s, because step happens during [0.5, 0.7] and 0.5 >= (end-start)/2 (contained in half of interval)
            if threshold is 60%:
            -> step is not in s, because 0.5 < 0.52 (contained in half of interval)

    Args:
        data: pandas dataframe containing steps.csv rows
        window_size:  amount of milliseconds [ms] in which to monitor activeness of steps
        total_size:  amount of seconds [s] in which to close the window of segments
        threshold: defines the percentage of overlap between steps and window segments
    Returns: windowed dataframe
    """
    num_window_segments = (total_size * 1000) // window_size  # default=25 ;
    columns = ['file', 'start', 'end', 'activities']
    d_new = pd.DataFrame(columns=columns)  # windowed entries go here
    running_sessions = data['file'].unique()  # get all running session of the current fold:

    # iterate over all running sessions:
    for session in tqdm(running_sessions, desc='Session', total=len(running_sessions)):
        total_elapsed = max(data[data['file'].str.contains(session)].elapsed)
        total_segments = math.ceil(total_elapsed * 1000 / window_size)
        session_steps = list(data[data['file'].str.contains(session)].elapsed)  # seconds, in order
        print(session_steps[:20])
        window_start = 0.0
        window_end = window_start + total_size
        segment_start = 0.000  # first recorded step always start at 0.0
        segment_end = segment_start + round(window_size / 1000, 4)  # adapt precision here
        normalized_segment_size = window_size / 1000
        window_activities = []

        for i in range(total_segments):

            print('=' * 20)
            print('start', segment_start, 'end', segment_end)
            # find all steps that occur within the current window segment
            steps_in_window = [step for step in session_steps if segment_start <= step <= segment_end]
            if len(steps_in_window) != 0:
                contained = is_contained([segment_start, segment_end],
                                         (steps_in_window[0] - 0.1), (steps_in_window[0] + 0.1),
                                         threshold=threshold)
                print('contained',contained)
                print('steps_in_window',steps_in_window)
                threshold_steps = [step for step in steps_in_window if (step % normalized_segment_size) / normalized_segment_size > threshold]
                print('threshold_steps',threshold_steps)

                print('threshold contained:', threshold_steps)
                # determine if the proportion of steps exceeds the threshold
                active = True if len(threshold_steps) > 0 else False
            else:
                active = False

            window_activities.append(active)

            # here instead of adding segmentwise, we add after current window is 'full'
            if len(window_activities) == num_window_segments:
                window_activities = [int(b) for b in window_activities]  # bool -> numerical
                entry = {'file': session, 'start': round(window_start, 4), 'end': round(window_end, 4),
                         'activities': window_activities}
                d_new = d_new.append(entry, ignore_index=True)
                assert d_new.shape[1] == 4, d_new.shape[1]  # check correct number of columns (redundant)
                window_activities = []
                window_start += 5.0
                window_end = window_start + total_size

            if i > 50:
                exit(0)

            # still track the 'old', individual segments on their own
            segment_start = segment_end
            segment_end = segment_start + round(window_size / 1000, 4)

    return d_new


def atomic_sed_window_concatenated(data, window_size=20, total_size=5.0):
    """
    Specialization to sed_window_concatenated with finer step labeling and greater label resolution.

    Logic is: smaller segments, and areas around active segments, get 'marked', according to the estimate of
    step duration of 200 milliseconds, meaning, with a window length of 20 milliseconds, the surrounding 8 segments of
    each active segment will be marked as active, as well as the active label itself.
    So, 200/window_size - 1 // 2 to the left or to the right, whenever there is no border.

    TODO: Looks like too many samples are annotated. Impacts melspectys.py

    Old(200ms segments):
                xyz,0.0,5.0,
                "[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]"

    Example: {'file': 'xyz', 'start': 0.0, 'end': 5.0,
                'activities':  [
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            }


    Args:
        data: pandas dataframe containing steps.csv rows
        window_size:  amount of milliseconds [ms] in which to monitor activeness of steps
        total_size:  amount of seconds [s] in which to close the window of segments
    Time per fold: 4022 s
    Returns: windowed dataframe

    """

    assert window_size < 200
    fixed_step_duration = 200  # AMOUNT OF MILLISECONDS, A STEP IS THOUGHT TO BE LONG
    activity_stretch = int(fixed_step_duration // window_size)  # AMOUNT OF SEGMENTS TO BE LABELLED AROUND STEPS
    if activity_stretch == 1:
        pass
    else:  # e.g. 200/20 = 10 -> 9 segments, 1 in step-occurence segment, 4 left and right (if there is no border)
        if activity_stretch % 2 == 0:
            activity_stretch -= 1  # 9 -> 8 for example

    num_window_segments = (total_size * 1000) // window_size  # default=250 for size 20 ;
    columns = ['file', 'start', 'end', 'activities']
    d_new = pd.DataFrame(columns=columns)  # windowed entries go here
    running_sessions = data['file'].unique()  # get all running session of the current set:
    first_run = True

    # iterate over all running sessions:
    for session in tqdm(running_sessions, desc='Session', total=len(running_sessions)):

        # acquire the total elapsed time [seconds] for the current session:
        total_elapsed = max(data[data['file'].str.contains(session)].elapsed)
        if first_run:
            print('\ntotal_elapsed:', total_elapsed)  # - 1247.482

        # calculate amount of segments:
        total_segments = math.ceil(total_elapsed * 1000 / window_size)
        if first_run:
            print('total_segments:', total_segments)  # - 6238 - [0.0, 0.2, 0.4, 0.6, 0.8, 1[s], ...

        # store step timestamps of this session:
        session_steps = list(data[data['file'].str.contains(session)].elapsed)  # seconds, in order
        if first_run:
            print(session_steps[
                  :20])  # - [0.0, 0.638, 1.989, 2.518, 3.04, 3.54, 3.94, 4.349, 4.749, 5.129, 5.528, 5.908, ...
        # session_steps = [int(s * 1000) for s in session_steps]  # milliseconds

        # now iterate over window, monitor step-occurances and label activeness:
        window_start = 0.0
        window_end = window_start + total_size

        segment_start = 0.000  # first recorded step always start at 0.0
        segment_end = segment_start + round(window_size / 1000, 4)  # adapt precision here
        if first_run:
            print('segment_start', segment_start, 'segment_end', segment_end)  # segment_start 0.0 segment_end 0.2

        # window_activities = []
        window_activities = np.zeros(int(num_window_segments))
        first_run = False
        counter = 0

        for i in range(total_segments):
            # check on atomic level (20ms) whether a step occured:
            t_occ = [i for i in session_steps if segment_start <= i <= segment_end]  # list for noise (multiple)
            active = True if len(t_occ) != 0 else False
            # if step occured fill segments around the step:
            if active:
                if activity_stretch == 1:
                    window_activities[counter] = int(active)
                else:
                    stretch_distance = activity_stretch//2
                    # check left border:
                    if (counter+1) - stretch_distance <= 0:
                        window_activities[counter:(counter + stretch_distance + 1)] = \
                            [int(active)] * (stretch_distance + 1)
                    # check right border:
                    elif (counter + stretch_distance) >= num_window_segments:
                        window_activities[(counter - stretch_distance):(counter+1)] = \
                            [int(active)] * (stretch_distance + 1)
                    # no border:
                    else:
                        # debug:
                        # print(counter)
                        # print(len(window_activities), (counter - stretch_distance), (counter + stretch_distance))
                        # print(window_activities[(counter - stretch_distance):(counter + stretch_distance + 1)])
                        # print([int(active)] * (2 * stretch_distance + 1))
                        window_activities[(counter - stretch_distance):(counter + stretch_distance + 1)] = \
                            [int(active)] * (2 * stretch_distance + 1)

            counter += 1
            # here instead of adding segmentwise, we add after current window is 'full'
            if (counter+1) == num_window_segments:
                entry = {'file': session, 'start': round(window_start, 4), 'end': round(window_end, 4),
                         'activities': window_activities}
                window_activities = np.zeros(int(num_window_segments))
                counter = 0
                d_new = d_new.append(entry, ignore_index=True)
                assert d_new.shape[1] == (4), d_new.shape[1]  # check correct number of columns (redundant)
                window_start += total_size
                window_end = window_start + total_size

            # still track the 'old', individual segments on their own
            segment_start = segment_end
            segment_end = segment_start + round(window_size / 1000, 4)


    return d_new


# ###################
# UTIL::
# ###################


def calc_minimal_step_dist(_loadpath):
    """
    Calculates the minimal, pair-wise distance between (all) occurances of steps.
    Based on the annotation type: 'elapsed'.
    Args:
        _loadpath: should contain all steps, e.g. steps.csv
    Returns: min (time) distance
    """

    loadpath = os.path.join(_loadpath, f'steps.csv')
    data = pd.read_csv(loadpath)
    min_d = 1.0
    for i, row in data.iterrows():
        if i + 1 == len(data):  # eof
            break
        if data.iloc[i + 1].elapsed == 0.0:  # new run
            continue
        dist = data.iloc[i + 1].elapsed - data.iloc[i].elapsed
        if dist < min_d:
            if dist == 0.0:  # noise
                continue
            print('new min: ', dist)
            min_d = dist
    # noise in data corrupts minimal distance.. some steps occur at the same time elapsed_t == elapsed_t+1
    return min_d


def calc_average_step_dist(_loadpath):
    """
    Calculates the average, pair-wise distance between (all) occurances of steps.
    Based on the annotation type: 'elapsed'.
    Args:
        _loadpath: should contain all steps, e.g. steps.csv
    Returns: avg (time) distance
    """
    loadpath = os.path.join(_loadpath, f'steps.csv')
    data = pd.read_csv(loadpath)
    avg = 0
    for i, row in data.iterrows():
        if i + 1 == len(data):  # eof
            break
        if data.iloc[i + 1].elapsed == 0.0:  # new run
            continue
        avg += data.iloc[i + 1].elapsed - data.iloc[i].elapsed
    avg = 1 / (len(data) - 1) * avg
    print(avg)  # 0.3894886873211401 ~= 400 ms
    return avg


def check_speaker_independancy(_loadpath):
    steppath = r'data/metadata/original/steps.csv'
    data = pd.read_csv(steppath)
    runs = data['file'].unique()
    print(len(runs))  # 197 annotated runs in steps.csv

    # RUNS CONTAINED IN SPLITS:
    splitsteppath = r'data/metadata/wdw5/fold0/fold0.csv'
    data = pd.read_csv(splitsteppath)
    print(len(data[
                  'file'].unique()))  # 188 annotated runs in fold0, fold1, fold2, ... This number is representative for our training.

    # RUNNERS:
    metap = 'data/metadata/original/meta.csv'
    meta = pd.read_csv(metap)
    # print(meta.head())

    root = 'data/kirun-data-by-runner/'
    dirs = os.listdir(root)
    runners = []
    for ses in dirs:
        path = os.path.join(root, ses)
        runners.extend(os.listdir(path))
    runners = set(runners)  # destroy duplicates

    usedrunners = pd.DataFrame()
    for runner in runners:
        usedrunners = usedrunners.append(meta[meta['runner'].str.contains(runner)], ignore_index=True)

    # sort by age descending:22

    print(usedrunners.sort_values(['Age']))
    print(np.mean(usedrunners['Experience']), np.max(usedrunners['Experience']),
          np.min(usedrunners['Experience']))  # 10.490196078431373, 53.0, 0.0
    print(np.mean(usedrunners['BMI']))  # 22.920272294955666
    print(np.mean(usedrunners['Weight']))  # 68.67450980392157

    # check speaker independancy:
    for i in range(5):
        fold = f'data/metadata/original/fold{i}'
        train = fold + '/steps_train.csv'
        test = fold + '/steps_test.csv'
        val = fold + '/steps_dev.csv'

        train = pd.read_csv(train)
        test = pd.read_csv(test)
        val = pd.read_csv(val)

        train0_runner = set(train['runner'].unique())
        test0_runner = set(test['runner'].unique())
        val0_runner = set(val['runner'].unique())

        print('train-test:',
              train0_runner.intersection(test0_runner))  # set() == empty set == none lie in the intersection
        print('train-val:',
              train0_runner.intersection(val0_runner))  # set() == empty set == none lie in the intersection

    fold = f'data/metadata/wdw5/combined'
    train = fold + '/train.csv'
    test = fold + '/test.csv'
    val = fold + '/devel.csv'

    train = pd.read_csv(train)
    test = pd.read_csv(test)
    val = pd.read_csv(val)

    print(len(train), len(test), len(val))

    train0_runner = set(train['file'].unique())
    test0_runner = set(test['file'].unique())
    val0_runner = set(val['file'].unique())

    print('train-test:',
          train0_runner.intersection(test0_runner))  # set() == empty set == none lie in the intersection
    print('train-val:',
          train0_runner.intersection(val0_runner))  # set() == empty set == none lie in the intersection
    print('train-train (test):',
          train0_runner.intersection(train0_runner))  # set() == empty set == none lie in the intersection


def alter_columns(path):
    d_train = pd.read_csv(path + '/train.csv')
    d_test = pd.read_csv(path + '/test.csv')
    d_val = pd.read_csv(path + '/devel.csv')
    d_fold = pd.read_csv(path + '/split.csv')

    d_train.start = d_train.start.round(1)
    d_train.end = d_train.end.round(1)
    d_test.start = d_test.start.round(1)
    d_test.end = d_test.end.round(1)
    d_val.start = d_val.start.round(1)
    d_val.end = d_val.end.round(1)
    d_fold.start = d_fold.start.round(1)
    d_fold.end = d_fold.end.round(1)

    d_train = d_train.loc[:, ~d_train.columns.str.contains('^Unnamed')]
    d_test = d_test.loc[:, ~d_test.columns.str.contains('^Unnamed')]
    d_val = d_val.loc[:, ~d_val.columns.str.contains('^Unnamed')]
    d_fold = d_fold.loc[:, ~d_fold.columns.str.contains('^Unnamed')]

    d_train.to_csv(path + f'/train1.csv')
    d_test.to_csv(path + f'/test1.csv')
    d_val.to_csv(path + f'/devel1.csv')
    d_fold.to_csv(path + f'/split1.csv')


def is_contained(segment_range, a, b, threshold):
    intersection = [max(segment_range[0], a), min(segment_range[1], b)]
    intersection_length = max(0, intersection[1] - intersection[0])
    interval_length = b - a
    return intersection_length >= threshold * interval_length

# ###################

def main():
    # LOAD ARGS:
    args = parse_args()
    loadpath = args.loadpath
    savepath = args.savepath
    window_size = args.window_size  # seconds for regression, milliseconds for sound event detection

    # DEFINE WINDOWING (careful for window sizes in args):

    # regression:
    #window_function = regression_window  # regression (default) ; adapt parameter for one leg windowing (in fct signature)
    window_function = segmented_regression_window  # regression

    # sound event detection:
    # window_function = sed_window
    # window_function = sed_segmented_window
    # window_function = sed_threshold_segmented_window
    # window_function = atomic_sed_window_concatenated

    # PERFORM WINDOWING:
    folds = [0]  # total: [0, 1, 2, 3, 4]
    create_windowed_folds(loadpath, savepath, window_function, window_size, folds=folds)  # given splits
    # create_new_windowed_split(loadpath, savepath, window_function, window_size, sessioncount=0.1)  # new splits


if __name__ == '__main__':
    main()
