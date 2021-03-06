#ifndef MILLIPYDE_WORKERS_H
#define MILLIPYDE_WORKERS_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "millipyde.h"

typedef void (*MPWorkItem)(void *arg);

typedef struct work_node {
    MPWorkItem work;
    void *arg;
    struct work_node *next;
} MPWorkNode;

typedef struct work_queue {
    MPWorkNode *head;
    MPWorkNode *tail;
    int len;
} MPWorkQueue;

typedef struct work_pool {
    MPWorkQueue queue;
    MPBool running;
    int num_threads;
    int num_threads_busy;
    pthread_mutex_t mux;
    pthread_cond_t work_available;
    pthread_cond_t working;
} MPDeviceWorkPool;



#ifdef __cplusplus
extern "C" {
#endif


MPStatus
mpwrk_create_work_node(MPWorkNode **result, MPWorkItem work, void *arg);

void 
mpwrk_destroy_work_node(MPWorkNode *node);

MPWorkNode *
mpwrk_work_queue_pop(MPDeviceWorkPool *work_pool);

MPStatus
mpwrk_work_queue_push(MPDeviceWorkPool *work_pool, MPWorkItem work, void *arg);

void
mpwrk_work_wait(MPDeviceWorkPool *work_pool);

void *
mpwrk_process_work(void *arg);

MPStatus
mpwrk_create_work_pool(MPDeviceWorkPool **result, int num_threads);

MPStatus
mpwrk_destroy_work_pool(MPDeviceWorkPool *work_pool);


#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_WORKERS_H