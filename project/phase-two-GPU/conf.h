#ifndef __CONF_H__
#define __CONF_H__

const int NUM_ROWS = 70000;
const int EPOCH_NUM = 1;
const double LEARNING_RATE = 0.1;

const int TRAIN = NUM_ROWS * 0.7;
const int TEST = NUM_ROWS - TRAIN;

const int BLOCK = 800;
const int CHUNCK = 32;

const int SCRATCH_SIZE = sizeof(double) * BLOCK * BLOCK;

#endif