/*
This code is based on John Schember's Thread pool implementation as
described in http://nachtimwald.com

The following is the acting license on his work:
*/

/*
Copyright John Schember <john@nachtimwald.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


#include <stdio.h>
#include <iostream>
#include <pthread.h>
#include "hip/hip_runtime.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "gpuarray.h"
#include "gpuimage.h"
#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_workers.h"
#include "millipyde_hip_util.h"

// TODO: make this more C++ friendly



extern "C"{


MPWorkNode *
mpwrk_create_work_node(MPWorkItem work, void *arg)
{
    MPWorkNode *node;

    if (!work)
    {
        return NULL;
    }

    node = (MPWorkNode *)malloc(sizeof(MPWorkNode));
    node->work = work;
    node->arg = arg;
    node->next = NULL;
    return node;
}


void mpwrk_destroy_work_node(MPWorkNode *node)
{
    if (node)
    {
        free(node);
    }
}


MPWorkNode *
mpwrk_work_queue_pop(MPDeviceWorkPool *work_pool)
{
    MPWorkNode *node;

    node = work_pool->queue.head;
    if (!node)
    {
        return NULL;
    }

    if (node->next == NULL)
    {
        work_pool->queue.head = NULL;
        work_pool->queue.tail = NULL;
    }
    else
    {
        work_pool->queue.head = node->next;
    }
    return node;
}


void
mpwrk_work_queue_push(MPDeviceWorkPool *work_pool, MPWorkItem work, void *arg)
{
    MPWorkNode *node;
    if (!work_pool)
    {
        return;
    }

    node = mpwrk_create_work_node(work, arg);
    if (!node)
    {
        return;
    }
    pthread_mutex_lock(&(work_pool->mux));

    if (work_pool->queue.head == NULL)
    {
        work_pool->queue.head = node;
        work_pool->queue.tail = work_pool->queue.head;
    }
    else
    {
        work_pool->queue.tail->next = node;
        work_pool->queue.tail = node;
    }

    pthread_cond_broadcast(&(work_pool->work_available));
    pthread_mutex_unlock(&(work_pool->mux));
}


void
mpwrk_work_wait(MPDeviceWorkPool *work_pool)
{
    pthread_mutex_lock(&(work_pool->mux));
    while (true)
    {
        // If running, wait for threads to finish. If stopped, wait for threads to die
        if ((work_pool->queue.head != NULL) ||
            (work_pool->running && work_pool->num_threads_busy != 0) ||
            (!work_pool->running && work_pool->num_threads != 0))
        {
            pthread_cond_wait(&(work_pool->working), &(work_pool->mux));
        }
        else {
            break;
        }
    }
    pthread_mutex_unlock(&(work_pool->mux));
}


void *
mpwrk_process_work(void *arg)
{
    MPDeviceWorkPool *work_pool = (MPDeviceWorkPool *)arg;
    MPWorkNode *node;

    while(true)
    {
        pthread_mutex_lock(&(work_pool->mux));

        while (work_pool->queue.head == NULL && work_pool->running)
        {
            pthread_cond_wait(&(work_pool->work_available), &(work_pool->mux));
        }

        if(!work_pool->running)
        {
            break;
        }
        
        node = mpwrk_work_queue_pop(work_pool);
        work_pool->num_threads_busy++;
        pthread_mutex_unlock(&(work_pool->mux));

        if (node != NULL)
        {
            node->work(node->arg);
            mpwrk_destroy_work_node(node);
        }

        pthread_mutex_lock(&(work_pool->mux));
        work_pool->num_threads_busy--;

        if (work_pool->running && work_pool->num_threads_busy == 0 && work_pool->queue.head == NULL)
        {
            pthread_cond_signal(&(work_pool->working));
        }

        pthread_mutex_unlock(&(work_pool->mux));
    }

    work_pool->num_threads--;
    pthread_cond_signal(&(work_pool->working));
    pthread_mutex_unlock(&(work_pool->mux));

    return NULL;
}


MPDeviceWorkPool * 
mpwrk_create_work_pool(int num_threads)
{
    MPDeviceWorkPool *work_pool;
    pthread_t thread;
    int i;

    work_pool = (MPDeviceWorkPool *)calloc(1, sizeof(MPDeviceWorkPool));
    work_pool->num_threads = num_threads;
    work_pool->running = MP_TRUE;
    
    pthread_mutex_init(&(work_pool->mux), NULL);
    pthread_cond_init(&(work_pool->work_available), NULL);
    pthread_cond_init(&(work_pool->working), NULL);

    work_pool->queue.head = NULL;
    work_pool->queue.tail = NULL;

    for(i = 0; i < num_threads; ++i)
    {
        pthread_create(&thread, NULL, mpwrk_process_work, work_pool);
        pthread_detach(thread);
    }

    return work_pool;
}


void
mpwrk_destroy_work_pool(MPDeviceWorkPool *work_pool)
{
    MPWorkNode *cur_node;
    MPWorkNode *next_node;

    if (work_pool == NULL)
    {
        return;
    }

    pthread_mutex_lock(&(work_pool->mux));
    cur_node = work_pool->queue.head;
    while(cur_node != NULL)
    {
        next_node = cur_node->next;
        mpwrk_destroy_work_node(cur_node);
        cur_node = next_node;
    }

    work_pool->running = MP_FALSE;
    pthread_cond_broadcast(&(work_pool->work_available));
    pthread_mutex_unlock(&(work_pool->mux));

    mpwrk_work_wait(work_pool);

    pthread_mutex_destroy(&(work_pool->mux));
    pthread_cond_destroy(&(work_pool->work_available));
    pthread_cond_destroy(&(work_pool->working));

    free(work_pool);
}




} // extern "C"