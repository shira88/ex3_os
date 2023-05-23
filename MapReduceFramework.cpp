#include "Barrier.h"
#include <pthread.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include "MapReduceFramework.h"

class ThreadContext
{
 public:
  int threadId;
  pthread_t pthread;

  const InputVec *inputVec;
  const MapReduceClient &client;
  std::atomic<int> *atomic_counter;

  Barrier *barrier{};
  pthread_mutex_t *mutex;

  IntermediateVec intermediateInnerVec;
  std::vector<IntermediateVec> *shuffled;
  OutputVec *outVec;

  ThreadContext (int threadId,
                 pthread_t pthread,
                 const InputVec *inputVec,
                 const MapReduceClient &client,
                 std::atomic<int> *atomic_counter,
//                 Barrier *barrier,
                 pthread_mutex_t *mutex,
                 std::vector<IntermediateVec> *shuffled,
                 OutputVec *outVec) :
      threadId (threadId),
      pthread (pthread),
      inputVec (inputVec),
      client (client),
      atomic_counter (atomic_counter),
//      barrier (barrier),
      mutex (mutex),
      shuffled (shuffled),
      outVec (outVec)
  {}

  ~ThreadContext () = default;
};

class MainThreadContext : public ThreadContext
{
 public:
  stage_t stage;
  ThreadContext **thread_contexts;
  int multiThreadLevel;
  bool isJobDone;

  MainThreadContext (int thread_id,
                     pthread_t pthread,
                     const InputVec *input_vec,
                     const MapReduceClient &client,
                     std::atomic<int> *atomic_counter,
//                     Barrier *barrier,
                     pthread_mutex_t *mutex,
                     std::vector<IntermediateVec> *shuffled,
                     OutputVec *out_vec,
                     stage_t stage,
                     ThreadContext **thread_contexts,
                     int multiThreadLevel,
                     bool isJobDone)
      : ThreadContext (thread_id,
                       pthread,
                       input_vec,
                       client,
                       atomic_counter,
//                       barrier,
                       mutex,
                       shuffled,
                       out_vec),
        stage (stage),
        thread_contexts (thread_contexts),
        multiThreadLevel (multiThreadLevel),
        isJobDone (isJobDone)
  {}

  ~MainThreadContext () = default;
};

K2 *get_max_value (MainThreadContext *mtc)
{
  K2 *max_value = nullptr;
  for (int i = 0; i < mtc->multiThreadLevel; i++)
  {
    if (!(mtc->thread_contexts[i]->intermediateInnerVec.empty ()))
    {
      if (max_value == nullptr)
      {
        max_value = mtc->thread_contexts[i]->intermediateInnerVec.back ()
            .first;
      }
      else
      {

        if (*max_value < *(mtc->thread_contexts[i]->intermediateInnerVec.back
            ().first))
        {
          max_value = mtc->thread_contexts[i]->intermediateInnerVec.back ()
              .first;
        }
      }
    }
  }

  return max_value;
}

template<typename K2> bool isEqual (const K2 *a, const K2 *b)
{
  return ! (*a < *b || *b < *a);
}

std::vector<IntermediateVec> *shuffle (MainThreadContext *mtc)
{
  auto vec = new std::vector<IntermediateVec> ();
  K2 *max_value = get_max_value (mtc);

  while (max_value != nullptr)
  {
    auto k_vec = new IntermediateVec ();

    for (int i = 0; i < mtc->multiThreadLevel; i++)
    {
      while ((!mtc->thread_contexts[i]->intermediateInnerVec.empty ()) &&
             (isEqual (mtc->thread_contexts[i]
                  ->intermediateInnerVec.back ().first, max_value)))
      {
        k_vec->push_back (mtc->thread_contexts[i]->intermediateInnerVec.back ());
        mtc->thread_contexts[i]->intermediateInnerVec.pop_back ();
      }
    }
    vec->push_back (*k_vec);
    (*(mtc->atomic_counter))++;
    max_value = get_max_value (mtc);
  }
  return vec;
}

template<typename K2, typename V2>
bool
comparePairs (const std::pair<K2 *, V2 *> &pair1, const std::pair<K2 *, V2 *> &pair2)
{
  // Compare the K2 values of the pairs
  return *(pair1.first) < *(pair2.first);
}

void *singleThread (void *arg)
{
  ThreadContext *tc = ((ThreadContext *) arg);

  if (tc->threadId == 0)
  {
    auto mtc = (MainThreadContext *) tc;
    mtc->stage = MAP_STAGE;
  }

  int old_value = tc->atomic_counter->fetch_add (1);

  auto pair = tc->inputVec->at (old_value);
  tc->client.map (pair.first, pair.second, tc);

  // sort
  std::sort (tc->intermediateInnerVec.begin (), tc->intermediateInnerVec.end (),
             comparePairs<K2, V2>);


    if (pthread_mutex_lock (tc->mutex) != 0)
    {
        fprintf (stderr, "system error: error on pthread_mutex_lock in cout before");
        exit (1);
    }

    std::cout << "before: " << tc->threadId << std::endl;

    if (pthread_mutex_unlock (tc->mutex) != 0)
    {
        fprintf (stderr, "system error: error on pthread_mutex_unlock in cout before");
        exit (1);
    }

  tc->barrier->barrier ();

    if (pthread_mutex_lock (tc->mutex) != 0)
    {
        fprintf (stderr, "system error: error on pthread_mutex_lock in cout after");
        exit (1);
    }

    std::cout << "after " << tc->threadId << std::endl;

    if (pthread_mutex_unlock (tc->mutex) != 0)
    {
        fprintf (stderr, "system error: error on pthread_mutex_unlock in cout after");
        exit (1);
    }

  if (tc->threadId == 0)
  {
    auto mtc = (MainThreadContext *) tc;
    mtc->stage = SHUFFLE_STAGE;
    tc->atomic_counter->store (0);
    mtc->shuffled = shuffle (mtc);
  }

  // TODO: it is recommended to use semaphore instead of barrier here
  tc->barrier->barrier ();

  // reduce
  if (tc->threadId == 0)
  {
    auto mtc = (MainThreadContext *) tc;
    mtc->stage = REDUCE_STAGE;
  }

  old_value = tc->atomic_counter->load () - 1;
  (*(tc->atomic_counter))--;
  auto reduce_vec = tc->shuffled->at (old_value);
  tc->client.reduce (&reduce_vec, tc);

  return 0;
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec,
                             OutputVec &outputVec,
                             int multiThreadLevel)
{
  // TODO: REMEMBER TO FREE ALL THE THREAD CONTEXTS, AND MAKE SURE THEY;RE
  //  DYNAMIC
  auto pthreads = new pthread_t[multiThreadLevel];
  auto contexts = new ThreadContext *[multiThreadLevel];

  std::vector<IntermediateVec> *shuffled;
  auto atomic_counter = new std::atomic<int> (0);

  pthread_mutex_t mutex (PTHREAD_MUTEX_INITIALIZER);
  stage_t stage = UNDEFINED_STAGE;

  auto *barrier = new Barrier (multiThreadLevel);

  MainThreadContext *job_handler;

  if (multiThreadLevel > 0)
  {
    job_handler = new MainThreadContext (0,
                                              pthreads[0],
                                              &inputVec,
                                              client,
                                              atomic_counter,
//                                              barrier,
                                              &mutex,
                                              shuffled,
                                              &outputVec,
                                              stage,
                                              contexts,
                                              multiThreadLevel,
                                              false);

    contexts[0] = job_handler;
    contexts[0]->barrier = barrier;
  }
  else
  {
    job_handler = nullptr;
  }

  for (int i = 1; i < multiThreadLevel; i++)
  {
    contexts[i] = new ThreadContext (i,
                                     pthreads[i],
                                     &inputVec,
                                     client,
                                     atomic_counter,
//                                     barrier,
                                     &mutex,
                                     shuffled,
                                     &outputVec);
      contexts[i]->barrier = barrier;
  }

  for (int i = 0; i < multiThreadLevel; i++)
  {
    pthread_create (pthreads + i, NULL, singleThread, *(contexts + i));
  }
  return job_handler;
}

void waitForJob (JobHandle job)
{
  auto mtc = (MainThreadContext *) job;
  if (!mtc->isJobDone)
  {
    for (int i = 0; i < mtc->multiThreadLevel; ++i)
    {
      pthread_join (mtc->thread_contexts[i]->pthread, NULL);
    }
  }
  mtc->isJobDone = true;
}

void getJobState (JobHandle job, JobState *state)
{
  auto mtc = (MainThreadContext *) job;
  stage_t stage = mtc->stage;
  float percentage = (float) mtc->atomic_counter->load () /
                     (float) mtc->multiThreadLevel;
  state->stage = stage;
  state->percentage = percentage;
}

void closeJobHandle (JobHandle job)
{
  //TODO DELETE EVERYTHING ACCORDING TO WHAT WE CHANGED, AND KNOW WHAT TO
  // FREE FROM THE THREAD CONTEXTS AND WHAT NOT

  waitForJob (job);
  auto mtc = (MainThreadContext *) job;
  std::cout << mtc->outVec->size() << std::endl;
//  delete[] mtc->thread_contexts;
//  delete mtc->barrier;
//  delete[] mtc->shuffled;
//  pthread_mutex_destroy (mtc->mutex);
}

void emit2 (K2 *key, V2 *value, void *context)
{
  auto intPair = IntermediatePair (key, value);
  auto tc = (ThreadContext *) context;
  tc->intermediateInnerVec.push_back (intPair);
}

void emit3 (K3 *key, V3 *value, void *context)
{
  // TODO: do we need to add the pairs in a specific order?

  auto outPair = OutputPair (key, value);
  auto tc = (ThreadContext *) context;
  if (pthread_mutex_lock (tc->mutex) != 0)
  {
    fprintf (stderr, "system error: error on pthread_mutex_lock in out vec");
    exit (1);
  }

  tc->outVec->push_back (outPair);

  if (pthread_mutex_unlock (tc->mutex) != 0)
  {
    fprintf (stderr, "system error: error on pthread_mutex_unlock in out vec");
    exit (1);
  }
}