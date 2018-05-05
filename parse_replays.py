"""
Modified version of pysc2.bin.replay_actions
https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_actions.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import signal
import sys
import threading
import time
import logging
import json
from collections import defaultdict

import numpy as np

from six.moves import queue
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller

from absl import app
from absl import flags
from pysc2.lib import gfile
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

from pysc2.bin.replay_actions import ReplayProcessor, ProcessStats, replay_queue_filler, valid_replay

from mappings import TEMPEST_UNITS, UNIT_ID_TO_NAME

FLAGS = flags.FLAGS

size = point.Point(16, 16)
interface = sc_pb.InterfaceOptions(
    raw=True, score=False,
    feature_layer=sc_pb.SpatialCameraSetup(width=24))
size.assign_to(interface.feature_layer.resolution)
size.assign_to(interface.feature_layer.minimap_resolution)

# Suppress some logging messages
logging.getLogger("absl").setLevel(logging.WARNING)


RESULT = {
    1: 'Win',
    2: 'Loss',
    3: 'Tie',
    4: 'Unknown'
}


class TempestReplayProcessor(ReplayProcessor):
    def run(self):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
        self._update_stage("spawn")
        replay_name = "none"
        while True:
            self._print("Starting up a new SC2 instance.")
            self._update_stage("launch")
            try:
                with self.run_config.start() as controller:
                    self._print("SC2 Started successfully.")
                    ping = controller.ping()
                    for _ in range(300):
                        try:
                            replay_path = self.replay_queue.get()
                        except queue.Empty:
                            self._update_stage("done")
                            self._print("Empty queue, returning")
                            return
                        try:
                            replay_name = os.path.basename(replay_path)[:10]
                            self.stats.replay = replay_name
                            self._print("Got replay: %s" % replay_path)
                            self._update_stage("open replay file")
                            replay_data = self.run_config.replay_data(replay_path)
                            self._update_stage("replay_info")
                            info = controller.replay_info(replay_data)
                            self._print("Processing Replay %s " % replay_name)
                            if valid_replay(info, ping):
                                self.stats.replay_stats.maps[info.map_name] += 1
                                for player_info in info.player_info:
                                    race_name = sc_common.Race.Name(
                                        player_info.player_info.race_actual)
                                    self.stats.replay_stats.races[race_name] += 1
                                map_data = None
                                if info.local_map_path:
                                    self._update_stage("open map file")
                                    map_data = self.run_config.map_data(info.local_map_path)

                                # Make directory to store output data
                                output_dir = "processed/{}".format(replay_name)
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)

                                metadata = {}
                                metadata['map_name'] = info.map_name
                                metadata['game_duration_loops'] = info.game_duration_loops
                                metadata['game_duration_seconds'] = info.game_duration_seconds
                                metadata['game_version'] = info.game_version
                                metadata['data_version'] = info.data_version
                                metadata['players'] = {}
                                races = []
                                total_apm = 0
                                total_mmr = 0

                                for p in info.player_info:
                                    player = {}
                                    race = sc_common.Race.Name(p.player_info.race_actual)
                                    races.append(race[0])
                                    player['race'] = race
                                    player['result'] = RESULT[p.player_result.result]
                                    player['apm'] = p.player_apm
                                    total_apm += p.player_apm
                                    player['mmr'] = p.player_mmr
                                    total_mmr += p.player_mmr
                                    metadata['players'][p.player_info.player_id] = player

                                metadata['matchup'] = '{}v{}'.format(min(races), max(races))
                                metadata['game_apm'] = total_apm // len(races)
                                metadata['game_mmr'] = total_mmr // len(races)

                                self._print("Matchup: {} @ {}".format(metadata['matchup'], metadata['map_name']))
                                self._print("Average MMR: {}".format(metadata['game_mmr']))
                                self._print("Game length: {}s".format(int(metadata['game_duration_seconds'])))

                                metadata_name = output_dir + '/metadata.json'
                                with open(metadata_name, 'w') as f:
                                    f.write(json.dumps(metadata, indent=4, sort_keys=True) + '\n')
                                    self._print("Wrote metadata to %s" % metadata_name)

                                for player_id in [1, 2]:
                                    self._print("Starting %s from player %s's perspective" % (
                                        replay_name, player_id))
                                    self.process_replay(controller, replay_data, map_data, player_id, output_dir, replay_name)
                            else:
                                self._print("Replay is invalid.")
                                self.stats.replay_stats.invalid_replays.add(replay_name)
                        except:
                            self._print("Unknown exception during replay {}".format(replay_name))
                            self.stats.replay_stats.invalid_replays.add(replay_name)
                        finally:
                            self.replay_queue.task_done()
                    self._update_stage("shutdown")
            except (protocol.ConnectionError, protocol.ProtocolError, remote_controller.RequestError):
                self.stats.replay_stats.crashing_replays.add(replay_name)
            except KeyboardInterrupt:
                return

    def process_replay(self, controller, replay_data, map_data, player_id, output_dir, replay_name):
        """Process a single replay, updating the stats."""
        self._update_stage("start_replay")
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        # feat = features.Features(controller.game_info())

        self.stats.replay_stats.replays += 1
        self._update_stage("step")
        controller.step()

        current_timestep = 1

        unit_data = []
        obs_data = []
        state_data = []

        cur_units = defaultdict(set)
        cur_obs = defaultdict(set)

        while True:
            self.stats.replay_stats.steps += 1
            self._update_stage("observe")
            obs = controller.observe()

            for u in obs.observation.raw_data.units:
                self.stats.replay_stats.unit_ids[u.unit_type] += 1

                # https://github.com/Blizzard/s2client-api/blob/master/include/sc2api/sc2_unit.h#L83
                # Alliance::Self == 1
                # Alliance::Ally == 2
                # Alliance::Neutral == 3
                # Alliance::Enemy == 4
                if u.alliance == 1:
                    unit_name = UNIT_ID_TO_NAME[u.unit_type]
                    tempest_id = TEMPEST_UNITS[unit_name]
                    cur_units[tempest_id].add(u.tag)

                elif u.alliance == 4:
                    unit_name = UNIT_ID_TO_NAME[u.unit_type]
                    tempest_id = TEMPEST_UNITS[unit_name]
                    cur_obs[tempest_id].add(u.tag)

            # https://github.com/deepmind/pysc2/blob/master/docs/environment.md#general-player-information
            res = obs.observation.player_common
            cur_state = (
                res.minerals, res.vespene, res.food_cap, res.food_used,
                res.food_army, res.food_workers, res.idle_worker_count,
                res.army_count, res.warp_gate_count, res.larva_count
            )

            # Collect unit observations over the course of 25 timesteps,
            # which is about 10 seconds of real in-game time.
            if current_timestep % 28 == 0 or obs.player_result:
                unit_count = [0 for _ in range(200)]
                obs_count = [0 for _ in range(200)]

                for unit_id, unit_tags in cur_units.items():
                    unit_count[unit_id] = len(unit_tags)

                for unit_id, unit_tags in cur_obs.items():
                    obs_count[unit_id] = len(unit_tags)

                unit_data.append(unit_count)
                obs_data.append(obs_count)
                state_data.append(cur_state)

                cur_units = defaultdict(set)
                cur_obs = defaultdict(set)

                if obs.player_result:
                    break

            current_timestep += 1
            self._update_stage("step")
            controller.step(FLAGS.step_mul)

        self._print("Total observations: {}".format(len(unit_data)))

        np_units = np.array(unit_data)
        fname = output_dir + '/player_{}_units.npy'.format(player_id)
        np.save(fname, np_units)

        np_obs = np.array(obs_data)
        fname = output_dir + '/player_{}_observed.npy'.format(player_id)
        np.save(fname, np_obs)

        np_state = np.array(state_data)
        fname = output_dir + '/player_{}_resources.npy'.format(player_id)
        np.save(fname, np_state)

        self._print("Wrote player {} data to {}.".format(player_id, output_dir))


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
