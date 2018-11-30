import os
import sys
import json
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation

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


def format_observation_data(data):
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


def format_prediction_data(data):
    """
    Returns X and Y for the given replay dictionary, where each
    matrix has 2n rows for a replay with n parsed timesteps.

    X_i: concatenation of (timestep, P1, P2) unit data at the current timestep
    Y_i: 1 if P1 won, else P2
    """
    replay = ReplayData(data)
    p1_won = replay.metadata['players']['1']['result'] == 'Win'

    X = []
    Y = []

    for i in range(replay.timesteps):
        for p1, p2 in ((replay.p1, replay.p2), (replay.p2, replay.p1)):
            x_i = np.array([i])
            x_i = np.append(x_i, p1.units[i])
            x_i = np.append(x_i, p2.units[i])
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

            # x, y = format_observation_data(data)
            x, y = format_prediction_data(data)
            if x.size != 0 and y.size != 0:
                Xs.append(x)
                Ys.append(y)
        else:
            bad_replays.add(entry.name)

    X = np.vstack(Xs)
    Y = np.hstack(Ys)

    # print("X.shape:", X.shape)
    # print("Y.shape:", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("X_train.shape:", X_train.shape)
    print("X_test.shape: ", X_test.shape)
    print("Y_train.shape:", Y_train.shape)
    print("Y_test.shape: ", Y_test.shape)

    num_good = sum(len(v) for k, v in good_replays.items())
    print("Found {}/{} good replays.".format(num_good, num_good + len(bad_replays)))
    print("Matchup breakdown:")
    for k, v in good_replays.items():
        print("{}: {}".format(k, len(v)))

    model = Sequential([
        Dense(32, input_dim=X_train.shape[1]),
        Activation('relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, validation_split=0.2, epochs=20, batch_size=10)
