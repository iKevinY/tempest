import os
import sys
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from load_data import ReplayData, get_replays
from mappings import RELEVANT_TEMPEST, RELEVANT_TEMPEST_IDS, TECH_BUILDINGS


SUPPLY_VALS = np.array([0] * 10 + [2, 4, 6, 6, 2, 3, 2, 4, 8, 2, 1, 3, 2, 2, 2, 5, 2, 2])

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

    # Index 4 is army supply under resources
    max_p1_army = np.argmax(replay.p1.resources[:, 4])
    max_p2_army = np.argmax(replay.p2.resources[:, 4])

    if max_p1_army <= 4 or max_p2_army <= 4:
        return None, None

    max_p1 = replay.p1.units[max_p1_army, RELEVANT_TEMPEST_IDS]
    max_p2 = replay.p2.units[max_p2_army, RELEVANT_TEMPEST_IDS]

    X.append(np.concatenate(([p1_apm, p1_mmr], max_p1)))
    Y.append(int(p1_won))

    X.append(np.concatenate(([p2_apm, p2_mmr], max_p2)))
    Y.append(int(not p1_won))

    return np.array(X), np.array(Y)


def supply_total(row):
    """
    Returns the unit count * supply value for each of the army units in a composition row.
    """
    return row * SUPPLY_VALS


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    max_replays = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only_matchup = sys.argv[3] if len(sys.argv) > 3 else None

    get_replays(data_path, max_replays, only_matchup, format_strategy_clustering_data)
