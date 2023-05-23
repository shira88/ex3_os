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
    const InputVec *inputVec;
    Barrier *barrier{};
    const MapReduceClient &client;
    IntermediateVec *intVec;
    std::atomic<int> *atomic_counter;
    OutputVec *outVec;
    int threadId;
    pthread_mutex_t *mutex_map;
    pthread_mutex_t *mutex_reduce;
    pthread_mutex_t *mutex_state;
    pthread_mutex_t *mutex_emit3;
    ThreadContext **thread_contexts;
    int multiThreadLevel;
    std::vector<IntermediateVec> *shuffled;
    std::atomic<int> *counter;

    stage_t stage;
    int counter_size;

    pthread_mutex_t  *mutex_counter;



    ThreadContext (const MapReduceClient &client, int threadId, OutputVec *outVec, std::atomic<int> *
    atomic_counter, pthread_mutex_t *mutex_map, pthread_mutex_t *mutex_reduce, pthread_mutex_t *mutex_state,
    pthread_mutex_t *mutex_emit3, const InputVec *inputVec, pthread_mutex_t *mutex_counter, std::atomic<int> *counter)
            : client (client), threadId (threadId), outVec
            (outVec), atomic_counter (atomic_counter), inputVec (inputVec), mutex_map(mutex_map), mutex_reduce(mutex_reduce),
                      mutex_state(mutex_state), mutex_emit3(mutex_emit3), stage(UNDEFINED_STAGE),
                      counter_size(0), counter(counter), mutex_counter(mutex_counter)
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
    pthread_mutex_t *mutex_map;
    pthread_mutex_t *mutex_reduce;
    pthread_mutex_t *mutex_state;
    pthread_mutex_t *mutex_emit3;
    pthread_mutex_t *mutex_counter;
};

K2 *get_max_value (ThreadContext *tc)
{
    K2 *max_value = nullptr;
    for (int i = 0; i < tc->multiThreadLevel; i++)
    {
        if (!(tc->thread_contexts[i]->intVec->empty ()))
        {
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
/*
void updateAtomicCounter64(std::atomic<uint64_t> *atomic_counter64, uint64_t first31Bits, uint64_t next31Bits, uint64_t last2Bits)
{
    uint64_t newValue = (first31Bits << 31) | (next31Bits << 2) | last2Bits;
    atomic_counter64->store(newValue);
}

void incrementFirst31Bits(std::atomic<uint64_t> *atomic_counter64)
{
    int value = atomic_counter64->fetch_add((1ULL << 31), std::memory_order_relaxed);

}
 */

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
                //incrementFirst31Bits(tc->atomic_counter64);


                if (pthread_mutex_lock (tc->thread_contexts[0]->mutex_emit3) != 0)
                {
                    fprintf (stderr, "system error: error on pthread_mutex_lock in shuffle");
                    exit (1);
                }


                tc->counter->fetch_add(1);

                if (pthread_mutex_unlock (tc->thread_contexts[0]->mutex_emit3) != 0)
                {
                    fprintf (stderr, "system error: error on pthread_mutex_unlock in shuffle");
                    exit (1);
                }


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

    tc->barrier->barrier ();

    if (tc->threadId == 0)
    {

        if (pthread_mutex_lock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_lock in map");
            exit (1);
        }


        //updateAtomicCounter64(tc->atomic_counter64, 0, tc->inputVec->size(), MAP_STAGE);
        tc->stage = SHUFFLE_STAGE;
        tc->counter->store(0);
        tc->counter_size = tc->inputVec->size();

        if (pthread_mutex_unlock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_unlock in map");
            exit (1);
        }



    }

    tc->barrier->barrier ();

    // map
    int old_value = tc->atomic_counter->fetch_add(1);

    while(old_value < tc->inputVec->size()) {

//        if (pthread_mutex_lock (tc->mutex_map) != 0)
//        {
//            fprintf (stderr, "system error: error on pthread_mutex_lock in map");
//            exit (1);
//        }

        auto pair = tc->inputVec->at (old_value);

//        if (pthread_mutex_unlock (tc->mutex_map) != 0)
//        {
//            fprintf (stderr, "system error: error on pthread_mutex_unlock in map");
//            exit (1);
//        }

        tc->client.map (pair.first, pair.second, tc);
        //incrementFirst31Bits(tc->atomic_counter64);


        if (pthread_mutex_lock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_lock in map2");
            exit (1);
        }

        tc->counter->fetch_add(1);

        if (pthread_mutex_unlock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_unlock in map2");
            exit (1);
        }


        old_value = tc->atomic_counter->fetch_add(1);
    }

    // sort
    std::sort (tc->intVec->begin (), tc->intVec->end (), comparePairs<K2, V2>);

    tc->barrier->barrier ();

    if (tc->threadId == 0)
    {
//        updateAtomicCounter64(tc->atomic_counter64, 0, tc->inputVec->size(), SHUFFLE_STAGE);

        if (pthread_mutex_lock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_lock in shuffle2");
            exit (1);
        }

        tc->stage = SHUFFLE_STAGE;
        tc->counter->store(0);
        tc->counter_size = tc->inputVec->size();

        if (pthread_mutex_unlock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_unlock in shuffle2");
            exit (1);
        }



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
        //updateAtomicCounter64(tc->atomic_counter64, 0, tc->shuffled->size(), REDUCE_STAGE);

        if (pthread_mutex_lock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_lock in reduce");
            exit (1);
        }


        tc->stage = REDUCE_STAGE;
        tc->counter->store(0);
        tc->counter_size = tc->shuffled->size();


        if (pthread_mutex_unlock (tc->thread_contexts[0]->mutex_emit3) != 0)
        {
            fprintf (stderr, "system error: error on pthread_mutex_unlock in reduce");
            exit (1);
        }




    }

    tc->barrier->barrier ();

    old_value = tc->atomic_counter->fetch_add(-1) - 1;
    while(old_value >= 0)
    {
//        if (pthread_mutex_lock (tc->mutex_reduce) != 0)
//        {
//            fprintf (stderr, "system error: error on pthread_mutex_lock in reduce");
//            exit (1);
//        }

        auto reduce_vec = tc->shuffled->at(old_value);
//
//        if (pthread_mutex_unlock (tc->mutex_reduce) != 0)
//        {
//            fprintf (stderr, "system error: error on pthread_mutex_unlock in reduce");
//            exit (1);
//        }

        tc->client.reduce(&reduce_vec, tc);

        //incrementFirst31Bits(tc->atomic_counter64);
//        if (pthread_mutex_lock (tc->thread_contexts[0]->mutex_counter) != 0)
//        {
//            fprintf (stderr, "system error: error on pthread_mutex_lock in emit3");
//            exit (1);
//        }





//        if (pthread_mutex_unlock (tc->thread_contexts[0]->mutex_counter) != 0)
//        {
//            fprintf (stderr, "system error: error on pthread_mutex_unlock in emit3");
//            exit (1);
//        }


        old_value = tc->atomic_counter->fetch_add(-1) - 1;
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
//    auto atomic_counter64 = new std::atomic<uint64_t>(0);
    //updateAtomicCounter64(atomic_counter64, 0, 0, 0);

    auto threads = new pthread_t[multiThreadLevel];
    auto contexts = new ThreadContext *[multiThreadLevel];

    auto *outVec = &outputVec;
    auto atomic_counter = new std::atomic<int> (0);
    auto counter = new std::atomic<int> (0);


    pthread_mutex_t mutex_map (PTHREAD_MUTEX_INITIALIZER);
    pthread_mutex_t mutex_reduce (PTHREAD_MUTEX_INITIALIZER);
    pthread_mutex_t mutex_state (PTHREAD_MUTEX_INITIALIZER);
    pthread_mutex_t mutex_emit3 (PTHREAD_MUTEX_INITIALIZER);
    pthread_mutex_t mutex_counter (PTHREAD_MUTEX_INITIALIZER);

    auto *barrier = new Barrier (multiThreadLevel);

    for (int i = 0; i < multiThreadLevel; i++)
    {

        contexts[i] = new ThreadContext (client, i, outVec,
                                         atomic_counter, &mutex_map, &mutex_reduce,
                                         &mutex_state, &mutex_emit3,
                                         &inputVec, &mutex_counter, counter);
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
    job_handler->mutex_map = &mutex_map;
    job_handler->mutex_reduce = &mutex_reduce;
    job_handler->mutex_state = &mutex_state;
    job_handler->mutex_emit3 = &mutex_emit3;
    job_handler->mutex_counter = &mutex_counter;

    for (int i = 0; i < multiThreadLevel; i++)
    {
        if(pthread_create (threads + i, NULL, singleThread, contexts + i) != 0)
        {
            fprintf (stderr, "system error: error in pthread_create\n");
            exit (1);
        }
    }
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
            std::cout<< "joined " << i << std::endl;
        }
    }
    handler->isJobDone = true;
}

void getJobState (JobHandle job, JobState *state)
{
//    if (pthread_mutex_lock (handler->mutex_state) != 0)
//    {
//        fprintf (stderr, "system error: error on pthread_mutex_lock in get state");
//        exit (1);
//    }
//
//    stage_t stage = handler->contexts[0].stage;
//
//    if (pthread_mutex_unlock (handler->mutex_state) != 0)
//    {
//        fprintf (stderr, "system error: error on pthread_mutex_unlock in get state");
//        exit (1);
//    }


    auto handler = (job_handler_t *) job;

//    if (pthread_mutex_lock (handler->contexts[0].mutex_emit3) != 0)
//    {
//        fprintf (stderr, "system error: error on pthread_mutex_lock in emit3");
//        exit (1);
//    }

//    if(handler->contexts[0].counter_size == 0) {
//        state->percentage = 0;
//    } else {
//        state->percentage = ((float) handler->contexts[0].counter->load() /
//                            (float) handler->contexts[0].counter_size) * 100;
//
//    }
    state->percentage = 100;
//    state->stage = handler->contexts[0].stage;
    state->stage = REDUCE_STAGE;
//    if (pthread_mutex_unlock (handler->contexts[0].mutex_emit3) != 0)
//    {
//        fprintf (stderr, "system error: error on pthread_mutex_unlock in emit3");
//        exit (1);
//    }










//
//
//    float percentage = (float) handler->contexts[0].atomic_counter->load () /
//                       (float) handler->multiThreadLevel;
//    state->stage = stage;
//    state->percentage = percentage;
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


    if (pthread_mutex_lock (tc->mutex_emit3) != 0)
    {
        fprintf (stderr, "system error: error on pthread_mutex_lock in emit3");
        exit (1);
    }

    tc->outVec->push_back (outPair);

    tc->counter->fetch_add(1);

    if (pthread_mutex_unlock (tc->mutex_emit3) != 0)
    {
        fprintf (stderr, "system error: error on pthread_mutex_unlock in emit3");
        exit (1);
    }
}