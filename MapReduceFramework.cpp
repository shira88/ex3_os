#include <pthread.h>
#include <array>
#include "MapReduceFramework.h"

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel)
{
  auto *thread_array = new pthread_t[multiThreadLevel];

}