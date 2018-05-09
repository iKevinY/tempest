import os
import sys


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


def has_complete_data(path):
    replay_files = set()
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                replay_files.add(entry.name)

    for fname in EXPECTED_FILES:
        if fname not in replay_files:
            return False

    return True



if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else '.'

    good_replays = set()
    bad_replays = set()

    with os.scandir(data_path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir():
                if has_complete_data(entry):
                    good_replays.add(entry.name)
                else:
                    bad_replays.add(entry.name)

    print("Found {}/{} good replays.".format(len(good_replays), len(good_replays) + len(bad_replays)))
