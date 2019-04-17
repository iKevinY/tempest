import os
import sys
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation

from load_data import ReplayData, load_parsed_replay
from mappings import RELEVANT_TEMPEST, RELEVANT_TEMPEST_IDS, TECH_BUILDINGS


def format_strategy_clustering_data(data):
    """
    Return the units belonging to the timestep right before the
    first army/building decrease of >5 for either player (indicating
    the strategy decided before an initial skirmish)

    Y_i is 1 if the player state represented in X_i won, else 0.
    """
    replay = ReplayData(data)
    p1_won = replay.metadata['players']['1']['result'] == 'Win'

    p1_apm = replay.metadata['players']['1']['apm']
    p1_mmr = replay.metadata['players']['1']['mmr']

    p2_apm = replay.metadata['players']['2']['apm']
    p2_mmr = replay.metadata['players']['2']['mmr']

    X = []
    Y = []

    max_p1 = 0
    max_p2 = 0

    for i in range(replay.timesteps):
        curr_p1 = replay.p1.units[i, RELEVANT_TEMPEST_IDS].sum()
        curr_p2 = replay.p2.units[i, RELEVANT_TEMPEST_IDS].sum()

        if curr_p1 < (max_p1 - 5) or curr_p2 < (max_p2 - 5):
            last_p1 = replay.p1.units[i - 1, RELEVANT_TEMPEST_IDS]
            last_p2 = replay.p2.units[i - 1, RELEVANT_TEMPEST_IDS]

            X.append(np.concatenate(([p1_apm, p1_mmr], last_p1)))
            Y.append(int(p1_won))

            X.append(np.concatenate(([p2_apm, p2_mmr], last_p2)))
            Y.append(int(not p1_won))

            return np.array(X), np.array(Y)

        else:
            max_p1 = max(max_p1, curr_p1)
            max_p2 = max(max_p2, curr_p2)

    return None, None



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
            if x is not None and y is not None:
                Xs.append(x)
                Ys.append(y)
            else:
                bad_replays.add(entry.name)
        else:
            bad_replays.add(entry.name)

    num_good = sum(len(v) for k, v in good_replays.items())
    print("Found {}/{} good replays.".format(num_good, num_good + len(bad_replays)))
    print("Matchup breakdown:")
    for k, v in good_replays.items():
        print("{}: {}".format(k, len(v)))

    X = np.vstack(Xs)
    Y = np.hstack(Ys)

    data = X[:, 2:]

    print("MMR distribution:", np.histogram(X[:, 1]))

    print("Clustering:")

    ap = AffinityPropagation(damping=0.7).fit(data)
    print(ap)
    print(ap.labels_)
    print("Total clusters: {}".format(ap.labels_.shape))

    for strat in data:
        print(list(strat[:len(TECH_BUILDINGS)]), list(strat[len(TECH_BUILDINGS):]))

    print([s[len("PROTOSS_"):] for s in RELEVANT_TEMPEST[:len(TECH_BUILDINGS)]])
    print([s[len("PROTOSS_"):] for s in RELEVANT_TEMPEST[len(TECH_BUILDINGS):]])
