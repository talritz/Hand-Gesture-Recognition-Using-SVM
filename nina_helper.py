def import_db2(folder_path, subject, rest_length_cap=999):
    """Function for extracting data from raw NinaiPro files for DB2.
    (Modified: Removed E3 block to save RAM and avoid FileNotFoundError)

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-40 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

    Returns:
        Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data
    """
    fs = 2000

    # --- Load Exercise 1 ---
    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E1_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    move = np.squeeze(np.array(data['restimulus']))

    # --- Load Exercise 2 ---
    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E2_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.vstack((emg, np.array(data['emg'])))
    rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    move_tmp = np.squeeze(np.array(data['restimulus']))
    move = np.append(move, move_tmp)  # Note no fix needed for this exercise

    # --- (E3 Loading Block Removed) ---

    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    
    # Safety check in case of empty regions
    if len(move_regions) == 0:
        return {'emg': emg, 'rep': rep, 'move': move, 'rep_regions': np.array([]), 'nb_capped': 0}

    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1
        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1

    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }