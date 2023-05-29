#include "Barrier.h"
#include <pthread.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include "MapReduceFramework.h"

#define LEAST_31_BITS 2147483647
#define MIDDLE_31_BITS 4611686016279904256

class ThreadContext
{
public:
    const MapReduceClient &client;
    int threadId;
    OutputVec *outVec;
    std::atomic<int> *atomic_counter;
    const InputVec *inputVec;
    Barrier *barrier{};
    IntermediateVec *intVec;
    pthread_mutex_t *output_vector_mutex;
    ThreadContext **thread_contexts;
    int multiThreadLevel;
    std::vector<IntermediateVec> *shuffled;
    std::atomic<uint64_t> *atomic_state;
    std::atomic<int> *atomic_emit2_counter;

    ThreadContext (const MapReduceClient &client,
                   int threadId,
                   OutputVec *outVec,
                   std::atomic<int> * atomic_counter,
                   pthread_mutex_t *output_vector_mutex,
                   const InputVec *inputVec,
                   std::atomic<uint64_t> *atomic_state,
                   std::atomic<int> *atomic_emit2_counter)
                   :
            client (client),
            threadId (threadId),
            outVec (outVec),
            atomic_counter (atomic_counter),
            inputVec (inputVec),
            output_vector_mutex(output_vector_mutex),
            atomic_state(atomic_state),
            atomic_emit2_counter(atomic_emit2_counter)
    {
        intVec = new IntermediateVec ();
    }

    ~ThreadContext () = default;
};

struct job_handler_t
{
    pthread_t *threads;
    ThreadContext **contexts;
    int multiThreadLevel;
    std::atomic<bool> *isJobDone;
    pthread_mutex_t *wait_for_job_mutex;
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


void change_state(std::atomic<uint64_t> *atomic_state, size_t size, stage_t stage)
{
    uint64_t val = (((uint64_t)size) << 31) + (((uint64_t)stage) << 62);
    atomic_state->store(val);
}

void inc_state(std::atomic<uint64_t> *atomic_state)
{
    atomic_state->fetch_add(1);
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
                   isEqual(tc->thread_contexts[i]->intVec->back ().first , max_value))
            {
                k_vec->push_back (tc->thread_contexts[i]->intVec->back ());
                tc->thread_contexts[i]->intVec->pop_back ();
                inc_state(tc->atomic_state);
            }
        }
        vec->push_back (*k_vec);
        tc->atomic_counter->fetch_add(1);
        max_value = get_max_value (tc);
    }
    return vec;
}

template<typename K2, typename V2>
bool comparePairs (const std::pair<K2 *, V2 *> &pair1, const std::pair<K2 *, V2 *> &pair2)
{
    return *(pair1.first) < *(pair2.first);
}

void *singleThread (void *arg)
{
    auto *tc = *static_cast<ThreadContext **>(arg);

    tc->barrier->barrier ();

    if (tc->threadId == 0)
    {
        change_state(tc->atomic_state, tc->inputVec->size(), MAP_STAGE);
    }

    tc->barrier->barrier ();

    // map
    int old_value = tc->atomic_counter->fetch_add(1);
    size_t input_size = tc->inputVec->size();

    while(old_value < input_size) {
        auto pair = tc->inputVec->at (old_value);

        tc->client.map (pair.first, pair.second, tc);
        inc_state(tc->atomic_state);
        old_value = tc->atomic_counter->fetch_add(1);
    }

    // sort
    std::sort (tc->intVec->begin (), tc->intVec->end (), comparePairs<K2, V2>);

    tc->barrier->barrier ();

    if (tc->threadId == 0)
    {
        change_state(tc->atomic_state, tc->atomic_emit2_counter->load(), SHUFFLE_STAGE);

        tc->atomic_counter->store (0);
        auto shuffled = shuffle (tc);

        for (int i = 0; i < tc->multiThreadLevel; i++)
        {
            tc->thread_contexts[i]->shuffled = shuffled;
        }
    }

    tc->barrier->barrier ();

    // reduce
    if (tc->threadId == 0)
    {
        change_state(tc->atomic_state, tc->shuffled->size(), REDUCE_STAGE);
    }

    tc->barrier->barrier ();

    old_value = tc->atomic_counter->fetch_add(-1) - 1;
    while(old_value >= 0)
    {
        auto reduce_vec = tc->shuffled->at(old_value);

        tc->client.reduce(&reduce_vec, tc);

        inc_state(tc->atomic_state);

        old_value = tc->atomic_counter->fetch_add(-1) - 1;
    }

    return 0;
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec,
                             OutputVec &outputVec,
                             int multiThreadLevel)
{
    auto threads = new pthread_t[multiThreadLevel];
    auto contexts = new ThreadContext *[multiThreadLevel];

    auto *outVec = &outputVec;
    auto atomic_counter = new std::atomic<int> (0);
    auto atomic_emit2_counter = new std::atomic<int> (0);

    auto atomic_state = new std::atomic<uint64_t> (0);

    pthread_mutex_t output_vector_mutex;
    if(pthread_mutex_init(&output_vector_mutex, nullptr) != 0)
    {
        std::cout << "System error: output_vector_mutex init" << std::endl;
        exit(1);
    }

    pthread_mutex_t wait_for_job_mutex;
    if(pthread_mutex_init(&wait_for_job_mutex, nullptr) != 0)
    {
        std::cout << "System error: wait_for_job_mutex init" << std::endl;
        exit(1);
    }

    auto *barrier = new Barrier (multiThreadLevel);

    for (int i = 0; i < multiThreadLevel; i++)
    {
        contexts[i] = new ThreadContext (client,
                                         i,
                                         outVec,
                                         atomic_counter,
                                         &output_vector_mutex,
                                         &inputVec,
                                         atomic_state,
                                         atomic_emit2_counter);
        contexts[i]->barrier = barrier;
        contexts[i]->thread_contexts = contexts;
        contexts[i]->multiThreadLevel = multiThreadLevel;
    }

    auto *job_handler = new job_handler_t ();
    job_handler->threads = threads;
    job_handler->contexts = contexts;
    job_handler->multiThreadLevel = multiThreadLevel;
    job_handler->isJobDone = new std::atomic<bool> (false);;
    job_handler->wait_for_job_mutex = &wait_for_job_mutex;

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
    auto handler = static_cast<job_handler_t *>(job);

//    if(pthread_mutex_lock(handler->wait_for_job_mutex) != 0)
//    {
//        std::cout << "System error: wait_for_job_mutex lock failed\n";
//        exit(1);
//    }

    if (!handler->isJobDone->load())
    {
        for (int i = 0; i < handler->multiThreadLevel; ++i)
        {
            pthread_join (handler->threads[i], NULL);
        }

        handler->isJobDone->store(true);
    }

//    if(pthread_mutex_unlock(handler->wait_for_job_mutex) != 0)
//    {
//        std::cout << "System error: wait_for_job_mutex unlock failed\n";
//        exit(1);
//    }
}

void getJobState (JobHandle job, JobState *state)
{
    auto handler = static_cast<job_handler_t *>(job);
    uint64_t val = handler->contexts[0]->atomic_state->load();
    auto stage = (stage_t)(val >> 62);
    auto size = (size_t)((MIDDLE_31_BITS & val) >> 31);
    int count = (int)(LEAST_31_BITS & val);

    if(size == 0)
    {
        state->percentage = 0;
    }
    else
    {
        state->percentage = ((float) count / (float) size) * 100;
    }

    state->stage = stage;
}

void closeJobHandle (JobHandle job)
{
    auto jobHandler = static_cast<job_handler_t*>(job);

    // we must call it because it's the only place where we do pthread_join, and we can't count on it to be called from the outside.
    waitForJob(jobHandler);

    delete[] jobHandler->threads;
    delete jobHandler->contexts[0]->shuffled;
    delete jobHandler->contexts[0]->atomic_counter;
    delete jobHandler->contexts[0]->atomic_emit2_counter;
    delete jobHandler->contexts[0]->atomic_state;
    delete jobHandler->contexts[0]->barrier;
    pthread_mutex_destroy(jobHandler->contexts[0]->output_vector_mutex);
    pthread_mutex_destroy(jobHandler->wait_for_job_mutex);

    size_t arr_size = jobHandler->contexts[0]->multiThreadLevel;
    for(size_t i = 0; i < arr_size; i++) {
        delete jobHandler->contexts[i];
    }

    delete[] jobHandler->contexts;
    delete jobHandler;
}

void emit2 (K2 *key, V2 *value, void *context)
{
    auto intPair = IntermediatePair (key, value);
    auto tc = static_cast<ThreadContext *>(context);

    tc->intVec->push_back (intPair);

    tc->atomic_emit2_counter->fetch_add(1);
}

void emit3 (K3 *key, V3 *value, void *context)
{
    auto outPair = OutputPair (key, value);
    auto tc = static_cast<ThreadContext *>(context);

    if(pthread_mutex_lock(tc->output_vector_mutex) != 0)
    {
        std::cout << "System error: output_vector_mutex lock failed\n";
        exit(1);
    }

    tc->outVec->push_back (outPair);

    if(pthread_mutex_unlock(tc->output_vector_mutex) != 0)
    {
        std::cout << "System error: output_vector_mutex unlock failed\n";
        exit(1);
    }
}