#ifndef MAPREDUCEFRAMEWORK_H
#define MAPREDUCEFRAMEWORK_H

#include "resources/MapReduceClient.h"

typedef void *JobHandle;

enum stage_t
{
    UNDEFINED_STAGE = 0, MAP_STAGE = 1, SHUFFLE_STAGE = 2, REDUCE_STAGE = 3
};

typedef struct
{
    stage_t stage;
    float percentage;
} JobState;

/*
 * The function receives as input intermediary element (K2, V2) and context
 * which contains data structure of the thread that created the intermediary
 * element. The function saves the intermediary element in the context data
 * structures. In addition, the function updates the number of intermediary
 * elements using atomic counter. Please pay attention that emit2 is called
 * from the client's map function and the context is passed from the
 * framework to the client's map function as parameter.
 */
void emit2 (K2 *key, V2 *value, void *context);

/*
 * The function receives as input output element (K3, V3) and context which
 * contains data structure of the thread that created the output element.
 * The function saves the output element in the context data structures
 * (output vector). In addition, the function updates the number of output
 * elements using atomic counter. Please pay attention that emit3 is called
 * from the client's reduce function and the context is passed from the
 * framework to the client's map function as parameter.

 */
void emit3 (K3 *key, V3 *value, void *context);

/*
 * This function starts running the MapReduce algorithm (with several
 * threads) and returns a JobHandle.
 *
 * client – The implementation of MapReduceClient or in other words the task
 * that the framework should run.
 *
 * inputVec – a vector of type std::vector<std::pair<K1*, V1*>>, the input
 * elements.
 *
 * outputVec – a vector of type std::vector<std::pair<K3*, V3*>>, to which the
 * output elements will be added before returning. You can assume that
 * outputVec is empty.
 *
 * multiThreadLevel – the number of worker threads to be used for running the
 * algorithm. You will have to create threads using c function
 * pthread_create. You can assume multiThreadLevel argument is valid
 * (greater or equal to 1).
 *
 * Returns - The function returns JobHandle that will be used for monitoring
 * the job.
 *
 * You can assume that the input to this function is valid
 */
JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel);

/*
 * a function gets JobHandle returned by startMapReduceFramework and waits
 * until it is finished.
 *
 * Hint – you should use the c function pthread_join.
 * It is legal to call the function more than once and you should handle it.
 * Pay attention that calling pthread_join twice from the same process has
 * undefined behavior and you must avoid that.
 */
void waitForJob (JobHandle job);

/*
 * this function gets a JobHandle and updates the state of the job into the
 * given JobState struct
 */
void getJobState (JobHandle job, JobState *state);

/*
 * Releasing all resources of a job. You should prevent releasing resources
 * before the job finished. After this function is called the job handle will
 * be invalid.
 * In case that the function is called and the job is not finished yet wait
 * until the job is finished to close it.
 * In order to release mutexes and semaphores (pthread_mutex, sem_t) you
 * should use the functions pthread_mutex_destroy, sem_destroy.
 */
void closeJobHandle (JobHandle job);

#endif //MAPREDUCEFRAMEWORK_H
