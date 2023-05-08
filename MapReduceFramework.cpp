#include "Barrier.h"
#include <pthread.h>
#include "MapReduceFramework.h"

struct ThreadContext {
    std::pair<K1*, V1*> pair;
    Barrier* barrier;
    MapReduceClient &client;
};

void* singleThread(void* arg) // TODO continue
{
  ThreadContext* tc = (ThreadContext*) arg;

  // map + sort
  tc->client.map (tc->pair.first, tc->pair.second, );

  tc->barrier->barrier();

  // reduce

  return 0;
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec,
                             OutputVec &outputVec,
                             int multiThreadLevel)
{
  auto threads = new pthread_t[multiThreadLevel];
  auto contexts = new ThreadContext[multiThreadLevel];

  auto barrier = new Barrier(multiThreadLevel);

  for (int i = 0; i < multiThreadLevel; ++i) {
    contexts[i] = {inputVec[i], barrier, client};
  }

  for (int i = 0; i < multiThreadLevel; ++i) {
    pthread_create(threads + i, NULL, singleThread, contexts + i);
  }

  for (int i = 0; i < multiThreadLevel; ++i) {
    pthread_join(threads[i], NULL);
  }

  delete[] threads;
  delete[] contexts;
  delete barrier;

}