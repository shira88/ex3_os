#include "Barrier.h"
#include <pthread.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include "MapReduceFramework_old.h"

class ThreadContext
{
public:
    const InputVec *inputVec;
    Barrier *barrier{};
    const MapReduceClient &client;
    IntermediateVec *intVec;
    std::atomic<int> *atomic_counter;
    OutputVec *outVec;
    int threadId;
    pthread_mutex_t *mutex;
    stage_t stage;
    ThreadContext **thread_contexts;
    int multiThreadLevel;
    std::vector<IntermediateVec> *shuffled;

    ThreadContext (const MapReduceClient &client, int threadId, OutputVec *outVec, std::atomic<int> *
    atomic_counter, pthread_mutex_t *mutex, stage_t stage, const InputVec *inputVec)
            : client (client), threadId (threadId), outVec
            (outVec), atomic_counter (atomic_counter), mutex (mutex), stage
                      (stage), inputVec (inputVec)
    {
        intVec = new IntermediateVec ();
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

K2 *get_max_value (ThreadContext *tc)
{
    K2 *max_value = nullptr;
    for (int i = 0; i < tc->multiThreadLevel; i++)
    {
        if (!(tc->thread_contexts[i]->intVec->empty ()))
        {
            /*
          if (max_value == nullptr
              | *max_value < *(tc->thread_contexts[i].intVec->back ().first))
          {
            max_value = tc->thread_contexts[i].intVec->back ().first;
          }
             */

            if(max_value == nullptr) {
                max_value = tc->thread_contexts[i]->intVec->back ().first;
            } else {

                if(*max_value < *(tc->thread_contexts[i]->intVec->back ().first)) {
                    max_value = tc->thread_contexts[i]->intVec->back ().first;
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

std::vector<IntermediateVec> *shuffle (ThreadContext *tc)
{
    auto vec = new std::vector<IntermediateVec> ();
    K2 *max_value = get_max_value (tc);

    while (max_value != nullptr)
    {
        auto k_vec = new IntermediateVec ();

        for (int i = 0; i < tc->multiThreadLevel; i++)
        {
            while ((!tc->thread_contexts[i]->intVec->empty ()) &&
                   isEqual(tc->thread_contexts[i]
                                   ->intVec->back ().first , max_value))
            {
                k_vec->push_back (tc->thread_contexts[i]->intVec->back ());
                tc->thread_contexts[i]->intVec->pop_back ();
            }
        }
        vec->push_back (*k_vec);
        tc->atomic_counter->fetch_add(1);
        max_value = get_max_value (tc);
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
    ThreadContext *tc = *((ThreadContext **) arg);

    if (tc->threadId == 0)
    {
        tc->stage = MAP_STAGE;
    }

    // map
    int old_value = tc->atomic_counter->fetch_add(1);

    while(old_value < tc->inputVec->size()) {

        auto pair = tc->inputVec->at (old_value);
        tc->client.map (pair.first, pair.second, tc);

        old_value = tc->atomic_counter->fetch_add(1);
    }



    // sort
    std::sort (tc->intVec->begin (), tc->intVec->end (), comparePairs<K2, V2>);

    tc->barrier->barrier ();

    if (tc->threadId == 0)
    {
        tc->stage = SHUFFLE_STAGE;
        tc->atomic_counter->store (0);
        auto shuffled = shuffle (tc);

        for (int i = 0; i < tc->multiThreadLevel; i++)
        {
            tc->thread_contexts[i]->shuffled = shuffled;
        }
    }

    // TODO: it is recommended to use semaphore instead of barrier here
    tc->barrier->barrier ();

    // reduce
    if (tc->threadId == 0)
    {
        tc->stage = REDUCE_STAGE;
    }

    old_value = tc->atomic_counter->fetch_add(-1) - 1;
    if(old_value >= 0)
    {
        auto reduce_vec = tc->shuffled->at(old_value);
        tc->client.reduce(&reduce_vec, tc);
    }

    return 0;
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec,
                             OutputVec &outputVec,
                             int multiThreadLevel)
{
    // TODO: REMEMBER TO FREE ALL THE THREAD CONTEXTS, AND MAKE SURE THEY;RE
    //  DYNAMIC
    auto threads = new pthread_t[multiThreadLevel];
    auto contexts = new ThreadContext *[multiThreadLevel];

    //NEED MULTIPE INTERMEDIATE VECTORS
    //auto *intVec = new IntermediateVec ();

    auto *outVec = &outputVec;
    auto atomic_counter = new std::atomic<int> (0);


    pthread_mutex_t mutex (PTHREAD_MUTEX_INITIALIZER);
    stage_t stage = UNDEFINED_STAGE;

    auto *barrier = new Barrier (multiThreadLevel);

    for (int i = 0; i < multiThreadLevel; i++)
    {

        contexts[i] = new ThreadContext (client, i, outVec,
                                         atomic_counter, &mutex, stage, &
                                                 inputVec);
        contexts[i]->barrier = barrier;
        contexts[i]->thread_contexts = contexts;
        contexts[i]->multiThreadLevel = multiThreadLevel;
    }


    auto *job_handler = new job_handler_t ();
    job_handler->threads = threads;
    job_handler->contexts = *contexts;
    job_handler->barrier = barrier;
    job_handler->outVec = outVec;
    job_handler->multiThreadLevel = multiThreadLevel;
    job_handler->isJobDone = false;
    job_handler->mutex = &mutex;


    for (int i = 0; i < multiThreadLevel; i++)
    {
//      auto thread = threads + i;
//      auto context = contexts + i;

        pthread_create (threads + i, NULL, singleThread, contexts + i);
    }


    /*
    auto *job_handler = new job_handler_t ();
    job_handler->threads = threads;
    job_handler->contexts = *contexts;
    job_handler->barrier = barrier;
    job_handler->outVec = outVec;
    job_handler->multiThreadLevel = multiThreadLevel;
    job_handler->isJobDone = false;
    job_handler->mutex = &mutex;
    */
    return job_handler;
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
    //TODO DELETE EVERYTHING ACCORDING TO WHAT WE CHANGED, AND KNOW WHAT TO
    // FREE FROM THE THREAD CONTEXTS AND WHAT NOT

    /*
    waitForJob (job);
    auto handler = (job_handler_t *) job;
    delete[] handler->threads;
    delete[] handler->contexts;
    delete handler->barrier;
    delete[] handler->outVec;
    pthread_mutex_destroy (handler->mutex);
     */
}

void emit2 (K2 *key, V2 *value, void *context)
{
    auto intPair = IntermediatePair (key, value);
    ThreadContext *tc = (ThreadContext *) context;
    tc->intVec->push_back (intPair);
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

    if (pthread_mutex_unlock (tc->mutex) != 0)
    {
        fprintf (stderr, "system error: error on pthread_mutex_unlock");
        exit (1);
    }
}