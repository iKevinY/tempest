# tempest

StarCraft II replay-processing pipeline.


## Installation

```
$ pip install -r requirements.txt
```


## Usage

PySC2 on Linux expects StarCraft II files to be located in `~/StarCraftII`. This
can be customized by setting the `SC2PATH` environment variable.

```sh
$ ./run.sh <path_to_replay_directory> [num_parallel_parsers]
$ ./train.sh [path_to_parsed_directory] [num_replays] [game_type (ie. TvT)]
```
