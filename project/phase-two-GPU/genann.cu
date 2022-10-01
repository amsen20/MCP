/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cu_utils.h"
#include "kernels.h"

#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

#define LOOKUP_SIZE 4096

double genann_act_hidden_indirect(const struct genann *ann, double a) {
    return ann->activation_hidden(ann, a);
}

double genann_act_output_indirect(const struct genann *ann, double a) {
    return ann->activation_output(ann, a);
}

const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;
double interval;
double lookup[LOOKUP_SIZE];

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif


double genann_act_sigmoid(const genann *ann unused, double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

void genann_init_sigmoid_lookup(const genann *ann) {
        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
        int i;

        interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
        }
}

double genann_act_sigmoid_cached(const genann *ann unused, double a) {
    assert(!isnan(a));

    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

double genann_act_linear(const struct genann *ann unused, double a) {
    return a;
}

double genann_act_threshold(const struct genann *ann unused, double a) {
    return a > 0;
}

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;


    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = (genann*)malloc(sizeof(genann));
    double *start;
    // allocate ann in device
    {
        ensure(cudaMalloc((void**)&start, size), "could not allocate in device.");
    }
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* Set pointers. */
    ret->weight = start;
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    genann_randomize(ret); // TODO

    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;

    // genann_init_sigmoid_lookup(ret);

    return ret;
}


void genann_randomize(genann *ann) {
    double *weight = (double*)malloc(sizeof(double) * ann->total_weights);
    for (int i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        weight[i] = r - 0.5;
    }
    ensure(cudaMemcpy(ann->weight, weight, sizeof(double) * ann->total_weights, cudaMemcpyHostToDevice), "could n't copy");
    free(weight);
}


void genann_free(genann *ann) {
    /* The weight, output, and delta pointers go to the same buffer. */
    ensure(cudaFree(ann->weight), "free ann");
    free(ann);
}

double *genann_run_parallel(genann *ann, double *inputs) {
    int w_size = ann->total_weights * sizeof(double);
    double *d_w = ann->weight;
    
    int o_size = ann->total_neurons * sizeof(double);
    double *d_i = ann->output;
    double *d_o = ann->output + ann->inputs;

    /* Copy the inputs to the scratch area, where we also store each neuron's
        * output, for consistency. This way the first layer isn't a special case. */
    cudaMemcpy(ann->output, inputs, sizeof(double) * ann->inputs, cudaMemcpyDeviceToDevice);

    double *sd_w = d_w, *sd_i = d_i;

    int h, j, k;

    /* Figure input layer */
    // matrix multiplied to vector
    mat_mul_vector_sig(
        ann->hidden,
        ann->inputs + 1,
        d_w,
        d_i,
        d_o
    );

    d_o += ann->hidden;
    d_w += ann->hidden * (ann->inputs + 1);
    d_i += ann->inputs;

    /* Figure hidden layers, if any. */
    for (h = 1; h < ann->hidden_layers; ++h) {
        // matrix multiplied to vector
        mat_mul_vector_sig(
            ann->hidden,
            ann->hidden + 1,
            d_w,
            d_i,
            d_o
        );

        d_o += ann->hidden;
        d_w += ann->hidden * (ann->hidden + 1);
        d_i += ann->hidden;
    }

    double *ret = ann->output + int(d_o - sd_i);

    /* Figure output layer. */
    // matrix multiplied to vector
    mat_mul_vector_sig(
        ann->outputs,
        ann->hidden + 1,
        d_w,
        d_i,
        d_o
    );

    d_o += ann->outputs;
    d_w += ann->outputs * (ann->hidden + 1);

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(d_w - sd_w == ann->total_weights);
    assert(d_o - sd_i == ann->total_neurons);

    return ret;
}


void genann_train_parallel(genann *ann, double *inputs, double *desired_outputs, double *scratch) {
    /* To begin with, we must run the network forward. */
    genann_run_parallel(ann, inputs);

    int w_size = ann->total_weights * sizeof(double);
    double *d_w = ann->weight;
    
    int o_size = ann->total_neurons * sizeof(double);
    double *d_o = ann->output;

    int del_size = (ann->total_neurons - ann->inputs) * sizeof(double);
    double *d_del = ann->delta;

    int des_size = ann->outputs * sizeof(double);
    double *d_des = desired_outputs;

    int h, j, k;

    /* First set the output layer deltas. */
    {
        double *o = d_o + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
        double *d = d_del + ann->hidden * ann->hidden_layers; /* First delta. */
        double *t = d_des; /* First desired output. */


        /* Set output layer deltas. */
        if (genann_act_output == genann_act_linear ||
                ann->activation_output == genann_act_linear) { // never gets here
            for (j = 0; j < ann->outputs; ++j) {
                exit(EXIT_FAILURE);
                *d++ = *t++ - *o++;
            }
        } else {
            // kernel for cost func
            diff_and_apply(t, o, d, ann->outputs);
        }
    }

    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first output and delta in this layer. */
        double *o = d_o + ann->inputs + (h * ann->hidden);
        double *d = d_del + (h * ann->hidden);

        /* Find first delta in following layer (which may be hidden or output). */
        double * dd = d_del + ((h+1) * ann->hidden);

        /* Find first weight in following layer (which may be hidden or output). */
        double * ww = d_w + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));

        // transpose w
        // matmul
        // cost func
        calc_delta(
            (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden),
            ann->hidden,
            ww,
            1,
            dd,
            d,
            o,
            scratch
        );
    }

    /* Train the outputs. */
    {
        /* Find first output delta. */
        double *d = d_del + ann->hidden * ann->hidden_layers; /* First output delta. */

        /* Find first weight to first output delta. */
        double *w = d_w + (ann->hidden_layers
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
                : (0));

        /* Find first output in previous layer. */
        double *i = d_o + (ann->hidden_layers
                ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
                : 0);

        /* Set output layer weights. */
        // update mat
        int step = (ann->hidden_layers ? ann->hidden : ann->inputs);
        upd_mat(
            ann->outputs,
            step + 1,
            w,
            d,
            i,
            1
        );
        w += (1 + step) * ann->outputs;
        assert(w - d_w == ann->total_weights);
    }

    /* Train the hidden layers. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first delta in this layer. */
        double *d = d_del + (h * ann->hidden);

        /* Find first input to this layer. */
        double *i = d_o + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);

        /* Find first weight to this layer. */
        double *w = d_w + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);


        int step = (h == 0 ? ann->inputs : ann->hidden);
        // update mat
        upd_mat(
            ann->hidden,
            step + 1,
            w,
            d,
            i,
            1
        );

    }
}

