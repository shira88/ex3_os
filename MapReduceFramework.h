#ifndef MAPREDUCEFRAMEWORK_H
#define MAPREDUCEFRAMEWORK_H

#include "resources/MapReduceClient.h"

typedef void* JobHandle;

enum stage_t {UNDEFINED_STAGE=0, MAP_STAGE=1, SHUFFLE_STAGE=2, REDUCE_STAGE=3};

typedef struct {
	stage_t stage;
	float percentage;
} JobState;

/*
 *
 */
void emit2 (K2* key, V2* value, void* context);

/*
 *
 */
void emit3 (K3* key, V3* value, void* context);

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
JobHandle startMapReduceJob(const MapReduceClient& client,
	const InputVec& inputVec, OutputVec& outputVec,
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
void waitForJob(JobHandle job);

/*
 *
 */
void getJobState(JobHandle job, JobState* state);

/*
 *
 */
void closeJobHandle(JobHandle job);
	
	
#endif //MAPREDUCEFRAMEWORK_H
