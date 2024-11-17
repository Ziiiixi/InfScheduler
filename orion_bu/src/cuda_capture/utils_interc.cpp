#include "intercept_temp.h"
#include <mutex>

std::mutex thread_ids_mutex;
// int get_idx() {

// 	// Each client thread has a unique ID in the scheduler.
// 	// Based on the thread id that is captured, find the proper index

// 	#ifdef SYS_gettid
// 		pid_t tid = syscall(SYS_gettid);
// 	#else
// 	#error "SYS_gettid unavailable on this system"
// 	#endif


// 		int idx = -1;
// 		int clients = *num_total_clients;
// 		int num_tids = 2*clients+1;

// 		for (int i=0; i<num_tids; i++) {
// 			if (tid == thread_ids[i]) {
// 				idx = i%(clients+1);
// 				break;
// 			}
// 		}
// 		if (idx == -1) {
// 			// set threads for backward pass
// 			// In PyTorch training, a different thread is used for the backward pass
// 			for (int i=clients+1; i<num_tids; i++) {
// 				if (thread_ids[i] == 0) {
// 					thread_ids[i] = tid;
// 					idx = i%(clients+1);
// 				}
// 			}
// 		}

// 		// set per-thread affinity
// 		int offset = 1;
// 		if (clients==2) {
// 			// for compatibility with AE experiments
// 			offset = 4;
// 		}
// 		if (idx > -1 && !affinity_set[idx]) {
// 			cpu_set_t  mask;
// 			CPU_ZERO(&mask);
// 			CPU_SET(idx+offset, &mask);
// 			int result = sched_setaffinity(0, sizeof(mask), &mask);
// 			assert (result==0);
// 			affinity_set[idx] = true;
// 		}
// 		return idx;
// }

int get_idx() {
    #ifdef SYS_gettid
        pid_t tid = syscall(SYS_gettid);
    #else
    #error "SYS_gettid unavailable on this system"
    #endif
    int idx = -1;
    int clients = *num_total_clients;
    int num_tids = 2 * clients + 1;

    // Protect access to thread_ids[]
    std::lock_guard<std::mutex> lock(thread_ids_mutex);

    // Look for tid in the thread_ids array
    for (int i = 0; i < num_tids; i++) {
        if (tid == thread_ids[i]) {
            idx = i % (clients + 1);
            break;
        }
    }
	// printf("idx is %d\n", idx);
    // If tid wasn't found, assign it to the next available slot for backward pass
    // if (idx == -1) {
    //     for (int i = clients + 1; i < num_tids; i++) {
    //         if (thread_ids[i] == 0) {
    //             thread_ids[i] = tid;
    //             idx = i % (clients + 1);
    //             break;
    //         }
    //     }
    // }
	// printf("idx is %d\n", idx);
    // Set per-thread affinity, if needed
    int offset = (clients == 2) ? 4 : 1; // Compatibility adjustment
    if (idx > -1 && !affinity_set[idx]) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(idx + offset, &mask);
        int result = sched_setaffinity(0, sizeof(mask), &mask);
        assert(result == 0);
        affinity_set[idx] = true;
    }
	// printf("idx is %d\n", idx);
    return idx;
}

void block(int idx, pthread_mutex_t** mutexes, queue<func_record>** kqueues) {

	// make sure all pending operations have completed
	while (1) {
		pthread_mutex_lock(mutexes[idx]);
		volatile int sz = kqueues[idx]->size();
		pthread_mutex_unlock(mutexes[idx]);
		if (sz==0)
			break;
	}

}