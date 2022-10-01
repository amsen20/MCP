#ifndef __CONF_H__
#define __CONF_H__

const int NUM_ROWS = 70000;
const int EPOCH_NUM = 1;
const double LEARNING_RATE = 0.1;

const int THREAD_NUM = 8;
const int TRAIN = (NUM_ROWS * 0.7) / THREAD_NUM * THREAD_NUM;
const int TEST = NUM_ROWS - TRAIN;

#endif