#!/usr/bin/env sh
python parse_replays.py --replays "$1" --parallel "${2:-1}"
