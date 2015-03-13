#!/bin/bash
set -ux

../med \
  --train_file 20news.train \
  --test_file 20news.test

