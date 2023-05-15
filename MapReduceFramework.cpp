#include "Barrier.h"
#include <pthread.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <array>
#include "MapReduceFramework.h"

class ThreadContext
{
 public:
  std::pair<K1 *, V1 *> pair;
  Barrier *barrier{};
  const MapReduceClient &client;
  IntermediateVec *intVec;
  std::atomic<int> *atomic_counter;
  OutputVec *outVec;
  int threadId;
  pthread_mutex_t *mutex;
  stage_t stage;

  ThreadContext (const MapReduceClient &client, IntermediateVec
  *intVec, int threadId, OutputVec *outVec, std::atomic<int> *
  atomic_counter, pthread_mutex_t *mutex, stage_t stage)
      : client (client), intVec (intVec), threadId (threadId), outVec
      (outVec), atomic_counter (atomic_counter), mutex (mutex), stage (stage)
  {
  }

  ~ThreadContext () = default;
};

struct job_handler_t
{
    pthread_t *threads;
    ThreadContext *contexts;
    Barrier *barrier;
    IntermediateVec *intVec;
    OutputVec *outVec;
    int multiThreadLevel;
    bool isJobDone;
    pthread_mutex_t *mutex;
};

void *singleThread (void *arg) // TODO continue
{
  ThreadContext *tc = (ThreadContext *) arg;

  if (tc->threadId == 0)
  {
    tc->stage = MAP_STAGE;
  }
  // map + sort
  tc->client.map (tc->pair.first, tc->pair.second, tc);

  tc->barrier->barrier ();

  if (tc->threadId == 0)
  {
    tc->stage = SHUFFLE_STAGE;
    shuffle ();
  }

  tc->barrier->barrier ();

  if (tc->threadId == 0)
  {
    tc->stage = REDUCE_STAGE;
  }

  // reduce
  tc->client.reduce (tc->intVec, tc);

  return 0;
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec,
                             OutputVec &outputVec,
                             int multiThreadLevel)
{
  auto threads = new pthread_t[multiThreadLevel];
  auto contexts = new ThreadContext *[multiThreadLevel];
  auto *intVec = new IntermediateVec ();
  auto *outVec = new OutputVec ();
  auto atomic_counter = new std::atomic<int> (0);
  pthread_mutex_t mutex (PTHREAD_MUTEX_INITIALIZER);
  stage_t stage = UNDEFINED_STAGE;

  for (int i = 0; i < multiThreadLevel; i++)
  {
    contexts[i] = new ThreadContext (client, intVec, i, outVec,
                                     atomic_counter, &mutex, stage);
  }

  auto *barrier = new Barrier (multiThreadLevel);

  for (int i = 0; i < multiThreadLevel; ++i)
  {
    contexts[i]->pair = inputVec[i];
    contexts[i]->barrier = barrier;
  }

  for (int i = 0; i < multiThreadLevel; ++i)
  {
    pthread_create (threads + i, NULL, singleThread, contexts + i);
  }

  auto *job_handler = new job_handler_t ();
  job_handler->threads = threads;
  job_handler->contexts = *contexts;
  job_handler->barrier = barrier;
  job_handler->intVec = intVec;
  job_handler->outVec = outVec;
  job_handler->multiThreadLevel = multiThreadLevel;
  job_handler->isJobDone = false;
  job_handler->mutex = &mutex;
  return job_handler_t;
}

void waitForJob (JobHandle job)
{
  auto handler = (job_handler_t *) job;
  if (!handler->isJobDone)
  {
    for (int i = 0; i < handler->multiThreadLevel; ++i)
    {
      pthread_join (handler->threads[i], NULL);
    }
  }
  handler->isJobDone = true;
}

void getJobState (JobHandle job, JobState *state)
{
  auto handler = (job_handler_t *) job;
  stage_t stage = handler->contexts[0].stage;
  float percentage = (float) handler->contexts[0].atomic_counter->load () /
                     (float) handler->multiThreadLevel;
  state->stage = stage;
  state->percentage = percentage;
}

void closeJobHandle (JobHandle job)
{
  waitForJob (job);
  auto handler = (job_handler_t *) job;
  delete[] handler->threads;
  delete[] handler->contexts;
  delete handler->barrier;
  delete[] handler->intVec;
  delete[] handler->outVec;
  pthread_mutex_destroy (handler->mutex);
}

void emit2 (K2 *key, V2 *value, void *context)
{
  auto intPair = IntermediatePair (key, value);
  ThreadContext *tc = (ThreadContext *) context;
  if (pthread_mutex_lock (tc->mutex) != 0)
  {
    fprintf (stderr, "system error: error on pthread_mutex_lock");
    exit (1);
  }

  tc->intVec->push_back (intPair);
  (*(tc->atomic_counter))++;

  if (pthread_mutex_unlock (tc->mutex) != 0)
  {
    fprintf (stderr, "system error: error on pthread_mutex_unlock");
    exit (1);
  }
}

void emit3 (K3 *key, V3 *value, void *context)
{
  // TODO: do we need to add the pairs in a specific order?

  auto outPair = OutputPair (key, value);
  ThreadContext *tc = (ThreadContext *) context;
  if (pthread_mutex_lock (tc->mutex) != 0)
  {
    fprintf (stderr, "system error: error on pthread_mutex_lock");
    exit (1);
  }

  tc->outVec->push_back (outPair);
  (*(tc->atomic_counter))++;

  if (pthread_mutex_unlock (tc->mutex) != 0)
  {
    fprintf (stderr, "system error: error on pthread_mutex_unlock");
    exit (1);
  }
}