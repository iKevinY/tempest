# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Modified version of pysc2.bin.replay_info
https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_info.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter

from pysc2 import run_configs

from absl import app
from pysc2.lib import gfile
from s2clientprotocol import common_pb2 as sc_common


def _replay_index(replay_dir):
    """Output information for a directory of replays."""
    run_config = run_configs.get()
    replay_dir = run_config.abs_replay_path(replay_dir)
    print("Checking: ", replay_dir)

    replay_counter = Counter()

    with run_config.start() as controller:
        bad_replays = []

        for i, file_path in enumerate(run_config.replay_paths(replay_dir)):
            file_name = os.path.basename(file_path)
            try:
                info = controller.replay_info(run_config.replay_data(file_path))
            except:
                bad_replays.append(file_name)
                continue

            if info.HasField("error"):
                # print("failed:", file_name, info.error, info.error_details)
                bad_replays.append(file_name)

            elif len(info.player_info) < 2:
                # print("less than 2 players in game:", file_name)
                bad_replays.append(file_name)

            else:
                mmr1 = info.player_info[0].player_mmr
                mmr2 = info.player_info[1].player_mmr

                if mmr1 < 1500 or mmr2 < 1500:
                    # print("invalid mmr:", file_name)
                    bad_replays.append(file_name)
                    continue

                p1 = sc_common.Race.Name(info.player_info[0].player_info.race_actual)[0]
                p2 = sc_common.Race.Name(info.player_info[1].player_info.race_actual)[0]

                game = '{}v{}'.format(min(p1, p2), max(p1, p2))
                replay_counter[game] += 1

            if i % 1000 == 0:
                print("Stats at replay #{}:".format(i), replay_counter)

        valid = sum(replay_counter.values())
        print("Processed {}/{} valid replays.".format(valid, valid + len(bad_replays)))
        print(replay_counter)


def main(argv):
    if not argv:
        raise ValueError("No replay directory or path specified.")
    if len(argv) > 2:
        raise ValueError("Too many arguments provided.")
    path = argv[1]

    try:
        if gfile.IsDirectory(path):
            return _replay_index(path)
    except KeyboardInterrupt:
        pass


def entry_point():  # Needed so the setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
