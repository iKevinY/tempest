import os
import sys
import json

import numpy as np


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



if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else '.'

    good_replays = set()
    bad_replays = set()

    with os.scandir(data_path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir():
                data = load_parsed_replay(entry)
                if data is not None:
                    good_replays.add(entry.name)
                else:
                    bad_replays.add(entry.name)

    print("Found {}/{} good replays.".format(len(good_replays), len(good_replays) + len(bad_replays)))
