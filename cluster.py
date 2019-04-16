import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from load_data import ReplayData, load_parsed_replay
from mappings import RELEVANT_TEMPEST_IDS


def format_strategy_clustering_data(data):
    """
    Returns X and Y for the given replay dictionary, where each row X_i
    are army + tech building at given timesteps (intended to represent
    what strategy a player is playing).

    Y_i is 1 if the player state represented in X_i won, else 0.
    """
    replay = ReplayData(data)
    p1_won = replay.metadata['players']['1']['result'] == 'Win'

    X = []
    Y = []

    for i in range(replay.timesteps):
        for p1, p2 in ((replay.p1, replay.p2), (replay.p2, replay.p1)):
            x_i = np.array([i])  # first column is timestep index
            x_i = np.append(x_i, p1.units[RELEVANT_TEMPEST_IDS])
            X.append(x_i)

            Y.append(1 if (p1_won and p1 is replay.p1) else 0)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y



if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    max_replays = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only_matchup = sys.argv[3] if len(sys.argv) > 3 else None

    print("Scanning through data directory for potential replays...")
    to_parse = []

    with os.scandir(data_path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir():
                to_parse.append(entry)

    print("Found {} potential replays.".format(len(to_parse)))
    good_replays = defaultdict(set)
    bad_replays = set()

    if max_replays is not None and max_replays > 0:
        print("Taking the first {} replays...".format(max_replays))
        to_parse = to_parse[:max_replays]

    Xs = []
    Ys = []

    for entry in tqdm(to_parse):
        data = load_parsed_replay(entry)
        if data is not None:
            matchup = data['metadata']['matchup']

            if only_matchup is not None and matchup != only_matchup:
                continue

            good_replays[matchup].add(entry.name)

            x, y = format_strategy_clustering_data(data)
            if x.size != 0 and y.size != 0:
                Xs.append(x)
                print(x)
                Ys.append(y)
        else:
            bad_replays.add(entry.name)

    num_good = sum(len(v) for k, v in good_replays.items())
    print("Found {}/{} good replays.".format(num_good, num_good + len(bad_replays)))
    print("Matchup breakdown:")
    for k, v in good_replays.items():
        print("{}: {}".format(k, len(v)))

    X = np.vstack(Xs)
    Y = np.hstack(Ys)
