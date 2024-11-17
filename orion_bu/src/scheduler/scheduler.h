#include <stdio.h>
#include <dlfcn.h>
#include <queue>
#include <vector>
#include <pthread.h>
#include <syscall.h>
#include <pwd.h>
#include <iostream>
#include <string.h>
#include <fstream>

#include "utils_sched.h"

//void* sched_func(void* args);

class Scheduler {

	public:
		void profile_prep(queue<func_record>** qbuffers, int num_clients, bool reef);
		void generate_partitions(int n, int start, std::vector<std::vector<int>>& partition, std::vector<std::vector<std::vector<int>>>& partitions) ;
		void profile_reset(int num_clients);
		std::pair<int, std::vector<std::vector<std::pair<int, int>>>>  computeTailLatencyForPartition(const std::vector<std::vector<int>>& partition, const std::vector<std::vector<float>>& latencyTable, int totalTPCs);
		std::pair<float, std::vector<std::pair<int, int>>> computeTailLatencyDP(const std::vector<int>& clients, const std::vector<std::vector<float>>& latencyTable, int tpc);
		void* busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq,  int depth, int hp_limit, int update_start);
		void execute_kernel_profile(int client_id, struct func_record frecord);
		// void* busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq,  int depth, int hp_limit, int update_start, int mymask);
		// void schedule_spacial(vector<func_record*> frecords, int num_clients, int depth);
		// void schedule_reef(vector<func_record*> frecords, int num_clients, int depth);
		// void schedule_KRISP(vector<func_record*> frecords, int num_clients, int depth);
		// void schedule_KRISP_I(vector<func_record*> frecords, int num_clients, int depth);
		// void schedule_KRISP_O(vector<func_record*> frecords, int num_clients, int depth);
		// int schedule_sequential(vector<func_record*> frecords, int num_clients, int start);

};

//void* sched_func(void* sched);
//Scheduler* sched_init();
