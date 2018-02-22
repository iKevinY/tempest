"""
Modified version of pysc2.bin.replay_actions
https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_actions.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import sys
import threading
import time
import logging

import numpy as np

from six.moves import queue
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point

from absl import app
from absl import flags
from pysc2.lib import gfile
from s2clientprotocol import sc2api_pb2 as sc_pb

from pysc2.bin.replay_actions import ReplayProcessor, ProcessStats, replay_queue_filler

from mappings import REAL_UNITS_IDS

FLAGS = flags.FLAGS

size = point.Point(16, 16)
interface = sc_pb.InterfaceOptions(
    raw=True, score=False,
    feature_layer=sc_pb.SpatialCameraSetup(width=24))
size.assign_to(interface.feature_layer.resolution)
size.assign_to(interface.feature_layer.minimap_resolution)

# Suppress some logging messages
logging.getLogger("absl").setLevel(logging.WARNING)


# https://github.com/Blizzard/s2client-api/blob/master/include/sc2api/sc2_gametypes.h
RACES = {
    1: "Terran",
    2: "Zerg",
    3: "Protoss"
}


class TempestReplayProcessor(ReplayProcessor):
    def process_replay(self, controller, replay_data, map_data, player_id):
        """Process a single replay, updating the stats."""
        self._update_stage("start_replay")
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        feat = features.Features(controller.game_info())

        self.stats.replay_stats.replays += 1
        self._update_stage("step")
        controller.step()

        current_timestep = 0

        unit_data = []

        while True:
            self.stats.replay_stats.steps += 1
            self._update_stage("observe")
            obs = controller.observe()

            for action in obs.actions:
                act_fl = action.action_feature_layer
                if act_fl.HasField("unit_command"):
                    self.stats.replay_stats.made_abilities[act_fl.unit_command.ability_id] += 1
                if act_fl.HasField("camera_move"):
                    self.stats.replay_stats.camera_move += 1
                if act_fl.HasField("unit_selection_point"):
                    self.stats.replay_stats.select_pt += 1
                if act_fl.HasField("unit_selection_rect"):
                    self.stats.replay_stats.select_rect += 1
                if action.action_ui.HasField("control_group"):
                    self.stats.replay_stats.control_group += 1

                try:
                    func = feat.reverse_action(action).function
                except ValueError:
                    func = -1
                self.stats.replay_stats.made_actions[func] += 1

            for valid in obs.observation.abilities:
                self.stats.replay_stats.valid_abilities[valid.ability_id] += 1

            curr_units = []

            for u in obs.observation.raw_data.units:
                self.stats.replay_stats.unit_ids[u.unit_type] += 1

                # unit.alliance == 1 indicates that it belongs to the player
                if u.alliance == 1 and u.unit_type in REAL_UNITS_IDS:
                    curr_units.append((current_timestep, u.unit_type, u.tag, u.pos.x, u.pos.y))

            for ability_id in feat.available_actions(obs.observation):
                self.stats.replay_stats.valid_actions[ability_id] += 1

            if obs.player_result:
                break

            self._update_stage("step")
            controller.step(FLAGS.step_mul)

            # Sort units by type, then by their unique ID (tag)
            for unit in sorted(curr_units):
                unit_data.append(unit)

            current_timestep += 1

        # TODO: Serialize this data to disk
        np_units = np.array(unit_data)
        print("Unit Data (shape: {})".format(np_units.shape))
        print(np_units)


def stats_printer(stats_queue):
    """A thread that consumes stats_queue and prints them every 10 seconds."""
    proc_stats = [ProcessStats(i) for i in range(FLAGS.parallel)]
    print_time = time.time()
    width = 107

    running = True
    while running:
        print_time += 10

        while time.time() < print_time:
            try:
                s = stats_queue.get(True, print_time - time.time())
                if s is None:  # Signal to print and exit NOW!
                    running = False
                    break
                proc_stats[s.proc_id] = s
            except queue.Empty:
                pass

        print(" Process stats ".center(width, "-"))
        print("\n".join(str(s) for s in proc_stats))
        print("=" * width)


def main(unused_argv):
    """Dump stats about all the actions that are in use in a set of replays."""
    run_config = run_configs.get()

    if not gfile.Exists(FLAGS.replays):
        sys.exit("{} doesn't exist.".format(FLAGS.replays))

    stats_queue = multiprocessing.Queue()
    stats_thread = threading.Thread(target=stats_printer, args=(stats_queue,))
    stats_thread.start()
    try:
        # For some reason buffering everything into a JoinableQueue makes the
        # program not exit, so save it into a list then slowly fill it into the
        # queue in a separate thread. Grab the list synchronously so we know there
        # is work in the queue before the SC2 processes actually run, otherwise
        # The replay_queue.join below succeeds without doing any work, and exits.
        print("Getting replay list:", FLAGS.replays)
        replay_list = sorted(run_config.replay_paths(FLAGS.replays))
        print(len(replay_list), "replays found.\n")
        replay_queue = multiprocessing.JoinableQueue(FLAGS.parallel * 10)
        replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                               args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        for i in range(FLAGS.parallel):
            p = TempestReplayProcessor(i, run_config, replay_queue, stats_queue)
            p.daemon = True
            p.start()
            time.sleep(1)  # Stagger startups, otherwise they seem to conflict somehow

        replay_queue.join()  # Wait for the queue to empty.
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")
    finally:
        stats_queue.put(None)  # Tell the stats_thread to print and exit.
        stats_thread.join()


if __name__ == '__main__':
    app.run(main)
