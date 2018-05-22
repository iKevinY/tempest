import os
import sys
import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm


class ReplayData:
    def __init__(self, data):
        self.metadata = data['metadata']
        self.timesteps = len(data['player_1_units'])
        self.p1 = PlayerData(
            data['player_1_units'],
            data['player_1_observed'],
            data['player_1_resources'],
        )

        self.p2 = PlayerData(
            data['player_2_units'],
            data['player_2_observed'],
            data['player_2_resources'],
        )


class PlayerData:
    def __init__(self, units, observed, resources):
        self.units = units
        self.observed = observed
        self.resources = resources


# All properly parsed replays should have these files present
EXPECTED_FILES = [
    'metadata.json',
    'player_1_units.npy',
    'player_1_observed.npy',
    'player_1_resources.npy',
    'player_2_units.npy',
    'player_2_observed.npy',
    'player_2_resources.npy',
]


def load_parsed_replay(path):
    expected = set(EXPECTED_FILES)
    replay_data = {}

    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file() and entry.name in expected:
                fname = os.path.join(path, entry.name)
                if entry.name.endswith('.npy'):
                    data = np.load(fname)
                elif entry.name.endswith('.json'):
                    with open(fname) as f:
                        data = json.load(f)
                else:
                    return None

                replay_data[entry.name.split('.')[0]] = data
                expected.remove(entry.name)

    if expected:
        return None
    else:
        return replay_data


def format_replay_data(data):
    """
    Returns X and Y for the given replay dictionary, where each
    matrix has 2(n - 1) rows for a replay with n parsed timesteps.

    X_i: concatenation of state, unit, and observed enemy unit data
    Y_i: ground truth of enemy unit count (from enemy perspective)
    """
    replay = ReplayData(data)

    X = []
    Y = []

    # Predict one timestep into the future, so n-1 total rows.
    for i in range(replay.timesteps - 1):
        for p1, p2 in ((replay.p1, replay.p2), (replay.p2, replay.p1)):
            x_i = p1.resources[i]
            x_i = np.append(x_i, p1.units[i])
            x_i = np.append(x_i, p1.observed[i])
            X.append(x_i)

            Y.append(replay.p2.units[i + 1])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y



if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else '.'

    print("Scanning through data directory for potential replays...")
    to_parse = []

    with os.scandir(data_path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir():
                to_parse.append(entry)

    print("Found {} potential replays.".format(len(to_parse)))
    good_replays = defaultdict(set)
    bad_replays = set()

    for entry in tqdm(to_parse):
        data = load_parsed_replay(entry)
        if data is not None:
            matchup = data['metadata']['matchup']
            format_replay_data(data)
            good_replays[matchup].add(entry.name)
        else:
            bad_replays.add(entry.name)

    num_good = sum(len(v) for k, v in good_replays.items())
    print("Found {}/{} good replays.".format(num_good, num_good + len(bad_replays)))
    print("Matchup breakdown:")
    for k, v in good_replays.items():
        print("{}: {}".format(k, len(v)))
