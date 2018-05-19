import os
import sys
import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm


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
    matrix has 2n rows for a replay with n parsed timesteps.

    X_i: concatenation of state, unit, and observed enemy unit data
    Y_i: TBD (potentially enemy unit count)
    """
    pass


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
            good_replays[matchup].add(entry.name)
        else:
            bad_replays.add(entry.name)

    num_good = sum(len(v) for k, v in good_replays.items())
    print("Found {}/{} good replays.".format(num_good, num_good + len(bad_replays)))
    print("Matchup breakdown:")
    for k, v in good_replays.items():
        print("{}: {}".format(k, len(v)))
