#!/usr/bin/env bash
#
# Author: Synrey Yee
#
# Created at: 05/07/2017
#
# Description: compile word2vec.c
#
# Last Modified at: 05/07/2017, by: Synrey Yee

gcc word2vec.c -o word2vec -lm -lpthread