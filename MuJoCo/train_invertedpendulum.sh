#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python src/run.py --config-path src/config_invertedpendulum_radial_ppo.json
