import os
import sys
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation

from load_data import ReplayData, load_parsed_replay
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


def which_strategy(row):
    """
    The input is a row which maps the total amount of the following units:

    Buildings:               Army:
    0: CYBERNETICSCORE       10: ADEPT             20: OBSERVER
    1: DARKSHRINE            11: ARCHON            21: ORACLE
    2: FLEETBEACON           12: CARRIER           22: PHOENIX
    3: FORGE                 13: COLOSSUS          23: SENTRY
    4: PHOTONCANNON          14: DARKTEMPLAR       24: STALKER
    5: ROBOTICSBAY           15: DISRUPTOR         25: TEMPEST
    6: ROBOTICSFACILITY      16: HIGHTEMPLAR       26: WARPPRISM
    7: STARGATE              17: IMMORTAL          27: ZEALOT
    8: TEMPLARARCHIVE        18: MOTHERSHIP
    9: TWILIGHTCOUNCIL       19: MOTHERSHIPCORE

    We want to categorize these counts into various strategies.
    """

    return 0


def main(data_path, max_replays, only_matchup):
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

    return X, Y


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    max_replays = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only_matchup = sys.argv[3] if len(sys.argv) > 3 else None

    main(data_path, max_replays, only_matchup)
