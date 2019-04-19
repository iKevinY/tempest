import os
import sys
from collections import defaultdict
from shutil import copytree

from tqdm import tqdm

from load_data import load_parsed_replay


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    max_replays = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only_matchup = sys.argv[3] if len(sys.argv) > 3 else 'PvP'

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

    copied = 0

    output_dir = 'processed_{}'.format(only_matchup.lower())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for entry in tqdm(to_parse):
        data = load_parsed_replay(entry)
        if data is not None:
            if data['metadata']['matchup'] == only_matchup:
                copytree('processed/{}'.format(entry.name), '{}/{}'.format(output_dir, entry.name))
                copied += 1

    print("Copied {} {} matchups to directory".format(copied, only_matchup))
