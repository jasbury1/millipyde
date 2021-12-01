#include <stdio.h>
#include <iostream>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

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


MPStatus
mpwrk_create_work_node(MPWorkNode **result, MPWorkItem work, void *arg)
{
    MPWorkNode *node;

    if (!work)
    {
        return WORK_ERROR_NULL_WORK_POOL;
    }

    node = (MPWorkNode *)malloc(sizeof(MPWorkNode));
    if (!node)
    {
        return WORK_ERROR_ALLOC_WORK_NODE;
    }

    node->work = work;
    node->arg = arg;
    node->next = NULL;

    *result = node;
    return MILLIPYDE_SUCCESS;
}


void 
mpwrk_destroy_work_node(MPWorkNode *node)
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


MPStatus
mpwrk_work_queue_push(MPDeviceWorkPool *work_pool, MPWorkItem work, void *arg)
{
    MPStatus ret_val;
    MPWorkNode *node;

    if (!work_pool)
    {
        // Return with no error. The pool might have just been stopped and deleted
        return MILLIPYDE_SUCCESS;
    }

    ret_val = mpwrk_create_work_node(&node, work, arg);
    if (ret_val != MILLIPYDE_SUCCESS)
    {
        return ret_val;
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

    return MILLIPYDE_SUCCESS;
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
            /*
            pid_t x = syscall(__NR_gettid);
            printf("TID %d\n", (int)x);
            */

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


MPStatus
mpwrk_create_work_pool(MPDeviceWorkPool **result, int num_threads)
{
    MPDeviceWorkPool *work_pool;
    pthread_t thread;
    int i;

    work_pool = (MPDeviceWorkPool *)calloc(1, sizeof(MPDeviceWorkPool));
    if (!work_pool)
    {
        return WORK_ERROR_ALLOC_WORK_POOL;
    }

    work_pool->num_threads = num_threads;
    work_pool->running = MP_TRUE;
    
    if (0 != pthread_mutex_init(&(work_pool->mux), NULL))
    {
        return WORK_ERROR_INIT_MUX;
    }
    if (0 != pthread_cond_init(&(work_pool->work_available), NULL))
    {
        return WORK_ERROR_INIT_COND;
    }
    if (0 != pthread_cond_init(&(work_pool->working), NULL))
    {
        return WORK_ERROR_INIT_COND;
    }

    work_pool->queue.head = NULL;
    work_pool->queue.tail = NULL;

    for(i = 0; i < num_threads; ++i)
    {
        pthread_create(&thread, NULL, mpwrk_process_work, work_pool);
        pthread_detach(thread);
    }

    *result = work_pool;
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpwrk_destroy_work_pool(MPDeviceWorkPool *work_pool)
{
    if (!work_pool)
    {
        // Not considered an error to destroy a pool that is NULL
        return MILLIPYDE_SUCCESS;
    }

    pthread_mutex_lock(&(work_pool->mux));

    MPWorkNode *cur_node;
    MPWorkNode *next_node;
    
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

    return MILLIPYDE_SUCCESS;
}




} // extern "C"