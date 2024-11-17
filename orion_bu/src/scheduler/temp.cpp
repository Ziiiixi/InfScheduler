#include "scheduler.h"
#include <vector>
#include <algorithm>
#include <cmath>
#define MIN_DURATION 100000 // might need to change this - emperical

using namespace std;
int priority_client = 1;
// globals
void* klib;
vector<vector<op_info>> op_info_vector;
int* fidx;
int* num_client_kernels;
int* num_client_max_iters;
int* num_client_cur_iters;

bool* locked;

std::chrono::time_point<std::chrono::high_resolution_clock>* client_starts;
std::chrono::time_point<std::chrono::high_resolution_clock>* total_client_starts;
bool** client_starts_set;
vector<vector<float>> client_durations;
int num_tpcs = 24;
int max_sms = 48; // v100
queue<struct func_record>** client_buffers;
pthread_mutex_t** client_mutexes;
queue<struct func_record>** buffers;
int* seen;
vector<int> client_progress;
vector<int> func_progress;
// cudnnHandle_t* global_handle0;
// cudnnHandle_t* global_handle1;

// fifo-globals
cudaStream_t sched_stream;
cudaStream_t sync_stream;
cudaEvent_t sched_event;

// profile-globals
cudaStream_t** sched_streams;
cudaStream_t** sync_streams;
cudaEvent_t*** events;
int* streams;
int* event_ids;
int status;
vector<int> max_sms_clients;
vector<bool> is_train;

// reef
int lp_idx = 0;
int penalty = 0;
bool** request_status;
bool* stops;
bool* stop_ack;

//spacial

vector<int> num_cur_clients;
bool *client_finished;
uint32_t mask = 0;
uint32_t *localMask;
vector<int> client_mask;
vector<int> client_block;
std::vector<std::vector<std::pair<int, int>>> current_schedule;
queue<int> q_client;
// std::vector<std::atomic<bool>> is_executing(5);
// bool *num_cur_clients_init = false;
std::vector<bool> is_executing;

void Scheduler::profile_reset(int num_clients) {

	for (int i=0; i<num_clients; i++) {
		seen[i] = 0;
		streams[i] = -1;
		fidx[i] = 0;
	}
}

void Scheduler::profile_prep(queue<func_record>** qbuffers, int num_clients, bool reef) {

	register_functions();
	client_buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		client_buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	int num = num_clients;

	sched_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));
	sync_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));

	for (int i=0; i<num; i++){
		sched_streams[i] = NULL;
		sync_streams[i] = NULL;
	}

	events = (cudaEvent_t***)malloc((num)*sizeof(cudaEvent_t**));
	for (int i=0; i<num; i++)
		events[i] = NULL;

	create_streams(sched_streams, num, reef);
	create_streams(sync_streams, num, reef);
	create_events(events, num);

	

	seen = (int*)calloc(num,sizeof(int));
	event_ids = (int*)calloc(num, sizeof(int));
	localMask = (uint32_t*)calloc(num,sizeof(uint32_t));
	
	streams = (int*)malloc(num_clients*sizeof(int));
	for (int i=0; i<num_clients; i++)
		streams[i] = -1;

	sched_stream = 0;

	status = -1;

}
void Scheduler::generate_partitions(int n, int start, std::vector<std::vector<int>>& partition, std::vector<std::vector<std::vector<int>>>& partitions) {
    // Base case: when start exceeds the number of kernels (n)
    if (start == n) {
        partitions.push_back(partition);  // Store the current partition
        return;
    }

    // Iterate through existing groups
    for (int i = 0; i < partition.size(); i++) {
        partition[i].push_back(start);  // Add kernel to existing group
        generate_partitions(n, start + 1, partition, partitions);
        partition[i].pop_back();  // Backtrack
    }

    // Create a new group
    partition.push_back({start});
    generate_partitions(n, start + 1, partition, partitions);
    partition.pop_back();  // Backtrack
}

std::pair<float, std::vector<std::pair<int, int>>> Scheduler::computeTailLatencyDP(const std::vector<int>& clients, const std::vector<std::vector<float>>& latencyTable, int tpc) {

	int n = clients.size();

    if(n == 1){
		// std::cout << "Client " << clients[0] << ": " << tpc << " TPCs" << std::endl;
		// printf("Final latency : %f\n", latencyTable[clients[0]][tpc-1]);
		std::vector<std::pair<int, int>> allocation = {{clients[0], tpc}}; 
        return {latencyTable[clients[0]][tpc - 1], allocation};
	}
    std::vector<std::vector<float>> dp(n + 1, std::vector<float>(tpc + 1, std::numeric_limits<float>::max()));
	std::vector<std::vector<int>> allocation(n + 1, std::vector<int>(tpc + 1, -1));

	for(int i = 0; i < tpc + 1; i++){
		dp[0][i] = 0.0f;
	}

    for (int i = 1; i <= n; ++i) {
        int client = clients[i - 1];
        
        for (int t = 1; t <= tpc; ++t) {
            for (int x = 1; x <= t; ++x) {
                if (latencyTable[client][x] != std::numeric_limits<float>::max()) {
					// printf("dp i %d, t %d, and value: %f, upper value: %f, latency table %f, i: %d, t: %d\n", i, t, dp[i][t], dp[i - 1][x], latencyTable[client][x],i,t);
                    float possibleLatency = std::max(dp[i - 1][t-x], latencyTable[client][x]);
                    if (possibleLatency < dp[i][t]) {
                        dp[i][t] = possibleLatency;
                        allocation[i][t] = x;
                    }
                }
            }
			// printf("-----------------\n");
        }
    }

    // std::cout << "DP Table:" << std::endl;
    // for (int i = 0; i <= n; ++i) {
    //     for (int j = 0; j <= tpc; ++j) {
    //         if (dp[i][j] == std::numeric_limits<float>::max()) {
    //             std::cout << "inf\t";  // Use 'inf' for infinity values
    //         } else {
    //             std::cout << dp[i][j] << "\t";
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    

    std::vector<std::pair<int, int>> tpcAllocation;
    int remainingTPCs = tpc;
    for (int i = n; i >= 1; --i) {
        int tpcForClient = allocation[i][remainingTPCs];
        tpcAllocation.push_back({clients[i - 1], tpcForClient});
        remainingTPCs -= tpcForClient;
    }

    // std::cout << "Optimal TPC allocation for each client:" << std::endl;
    // for (const auto& alloc : tpcAllocation) {
    //     std::cout << "Client " << alloc.first << ": " << alloc.second << " TPCs" << std::endl;
    // }

    return {dp[n][tpc], tpcAllocation};
}

std::pair<int, std::vector<std::vector<std::pair<int, int>>>> Scheduler::computeTailLatencyForPartition(
    const std::vector<std::vector<int>>& partition,
    const std::vector<std::vector<float>>& latencyTable,
    int totalTPCs
) {

    std::vector<std::vector<std::pair<int, int>>> tpcAllocation;  // Store TPC allocation for each group
    int tpcsPerBatch = totalTPCs;
    float totalLatency = 0;
	
    for (int i = 0; i < partition.size(); ++i) {
        const std::vector<int>& group = partition[i];
        int groupTPCs = tpcsPerBatch;
        auto result = computeTailLatencyDP(group, latencyTable, groupTPCs);
		totalLatency += result.first;
		tpcAllocation.push_back(result.second);
    }
    return {totalLatency, tpcAllocation};
}
void Scheduler::execute_kernel_profile(int client_id, struct func_record frecord) {
	// volatile int sz1 = client_buffers[client_id]->size();
	// 				printf("sz1 is %d\n", sz1);
    schedule_kernel(frecord, sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
    pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
	// volatile int sz2 = client_buffers[client_id]->size();
	// 				printf("sz2 is %d\n", sz2);
	is_executing[client_id] = false;
}


//Multi threading

// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

	
// 	// int num_all_clients = num_clients;
// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				if(frecords[i]!=NULL &&  client_progress[i] > 0 && num_client_cur_iters[i]>9){
// 					unsetmask_nomutex(i);
// 				}
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				int kernel_idx = seen[j];
// 				op_info op_info_cur = op_info_vector[j][kernel_idx];
// 				if(num_client_cur_iters[j] > 9){
// 					if (is_executing[j] == false) 
// 					{	
// 						is_executing[j] = true;
// 						std::thread profile_thread(&Scheduler::execute_kernel_profile, this, j, *(frecords[j]));
// 						profile_thread.detach();
// 						continue;

// 					}
// 				}
// 				else{
// 					// Regular kernel scheduling (non-profiled) and pop from queue in the main thread
// 					// volatile int sz1 = client_buffers[j]->size();
// 					// printf("sz1 is %d\n", sz1);
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
			
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 					// num_all_clients-=1;
// 					unsetmask(client_mutexes[i], i);
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// unsetmask(client_mutexes[i], i);
// 					// num_all_clients +=1;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }


// for profile
void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
	int start0 = 0;
	int start1 = 0;

	int prev_large = -1;
	int hp_running = -1;

	bool inf_finished = false;
	bool started = false;
 	std::chrono::time_point<std::chrono::system_clock> start_time;
	auto start_total = std::chrono::high_resolution_clock::now();

	vector<bool> total_client_set(num_clients, false);
	vector<int> profiles(num_clients, -1);
	vector<int> cur_sms(num_clients, -1);
	// int hp_client = num_clients-1;

	int hp_client = 1;
	int lp_client = 0;

	bool large_found = false;
	long sum = 0; // sum of durations of ongoing BE kernels
	long size = 0; // sum of sizes of in-the-queues BE kernels
	int start = -1;

	// BS - works only for 2 clients for now
	// TODO: check this
	int low_sms = 0;
	int high_sms = max_sms_clients[0]; // 0 is the lp client
	int sm_threshold = max_sms_clients[0]/2;
	float hp_iter_duration = 0.0; // 1 is the hp client
	float hp_limit_float = (float)hp_limit;

	
	// int num_all_clients = num_clients;
	while(1) {

		if (!is_train[hp_client]) {
			sm_threshold = max_sms;
			update_start = INT_MAX;
		}
		
		vector<func_record*> frecords(num_clients, NULL);
		size = 0;

		for (int i=0; i<num_clients; i++) {

			if (seen[i] == num_client_kernels[i])
				continue;

			pthread_mutex_lock(client_mutexes[i]);
			volatile int sz = client_buffers[i]->size();
			if (sz > 0) {
				frecords[i] = &(client_buffers[i]->front());
				if(frecords[i]!=NULL &&  client_progress[i] > 0 && num_client_cur_iters[i]>9){
					unsetmask_nomutex(i);
				}
				int cur_iter = num_client_cur_iters[i];
				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					client_starts[i] = std::chrono::high_resolution_clock::now();
					client_starts_set[i][cur_iter] = true;
					if (!total_client_set[i]) {
						total_client_starts[i] = std::chrono::high_resolution_clock::now();
						total_client_set[i] = true;
					}
				}
			}
			pthread_mutex_unlock(client_mutexes[i]);
		}


		// set<int> scheduled_clients;	
		// set<int> unscheduled_clients;
		// while (!q_client.empty()) {
		// 	int client_idx = q_client.front();
		// 	q_client.pop();
		// 	int kernel_idx = seen[client_idx];
		// 	op_info op_info_cur = op_info_vector[client_idx][kernel_idx];
		// 	// printf("2 number of tpcs: %d\n", num_tpcs);
		// 	if (num_tpcs >= op_info_cur.Knee_TPC && frecords[client_idx] != NULL) {
		// 		scheduled_clients.insert(client_idx);
		// 		setmask(client_mutexes[client_idx], op_info_cur.Knee_TPC, client_idx);
		// 		std::cout << "Name of kernel: " << op_info_cur.name
		// 				<< " | Client ID: " << client_idx
		// 				<< " | Blocks: " << client_block[client_idx]
		// 				<< " | TPC Usage: " << op_info_cur.Knee_TPC
		// 				<< " | Critical: " << op_info_cur.Is_Critical
		// 				<< " | Iteration: " << num_client_cur_iters[client_idx]
		// 				<< " | Kernel Index: " << client_progress[client_idx]
		// 				<< " | TPC Usage: " << op_info_cur.Knee_TPC
		// 				<< " | Knee TPC: " << op_info_cur.Knee_TPC
		// 				<< std::endl;
		// 		client_block[client_idx] = 0;
		// 		int j = client_idx;
		// 		client_progress[client_idx] += 1;
		// 		schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
		// 		pop_from_queue(client_buffers[j], client_mutexes[j], j);
		// 	} else {
		// 		printf("try to schedule for client %d but no success, cur number of tpcs: %d\n", client_idx, num_tpcs);
		// 		client_block[client_idx] +=1;
		// 		unscheduled_clients.insert(client_idx);
		// 	}
		// }

		// for (auto it = unscheduled_clients.rbegin(); it != unscheduled_clients.rend(); ++it) {
		// 	q_client.push(*it);
		// }



		
		for (int j=0; j<num_clients; j++) {
			if(scheduled_clients.count(j) > 0 || unscheduled_clients.count(j) > 0){
				continue;
			}
			if (frecords[j] != NULL) {
				int kernel_idx = seen[j];
				op_info op_info_cur = op_info_vector[j][kernel_idx];
				// printf("num_all clients %d\n", num_all_clients);
				if(num_client_cur_iters[j]>9){
					if (frecords[j]->type != MALLOC_RECORD && 
						frecords[j]->type != MEMCPY_RECORD && 
						frecords[j]->type != MEMSET_RECORD && 
						frecords[j]->type != FREE_RECORD) 
						{
							if (num_tpcs > 0) {
								int tpc_usage = 0; 
								if (op_info_cur.Is_Critical == 1) {
									if (num_tpcs < op_info_cur.Knee_TPC) {
										// printf("try to schedule for client %d, cur number of tpcs: %d\n", j, num_tpcs);
										// q_client.push(j);
										client_block[j] +=1;
										continue;  
									}
									tpc_usage = op_info_cur.Knee_TPC; 
								} else {
									tpc_usage = std::min(num_tpcs, op_info_cur.Knee_TPC);
								}

								setmask(client_mutexes[j], tpc_usage, j);
								std::cout << "Name of kernel: " << op_info_cur.name
										<< " | Client ID: " << j
										<< " | Blocks: " << client_block[j]
										<< " | TPC Usage: " << op_info_cur.Knee_TPC
										<< " | Critical: " << op_info_cur.Is_Critical
										<< " | Iteration: " << num_client_cur_iters[j]
										<< " | Kernel Index: " << client_progress[j]
										<< " | TPC Usage: " << tpc_usage
										<< " | Knee TPC: " << op_info_cur.Knee_TPC
										<< std::endl;
								client_progress[j] += 1;
							}
							else{
								continue;
							}
						}
						client_block[j] = 0;
						schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
						pop_from_queue(client_buffers[j], client_mutexes[j], j);
				}
				else{
					// if (frecords[j]->type != MALLOC_RECORD && 
					// 	frecords[j]->type != MEMCPY_RECORD && 
					// 	frecords[j]->type != MEMSET_RECORD && 
					// 	frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) 
					// 	{
					// 		// auto start_time = std::chrono::high_resolution_clock::now();
					// 		// auto start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();
					// 		// client_progress[j] += 1;
					// 		// std::cout << "Name of kernel: " << op_info_cur.name
					// 		// 		<< " | Client ID: " << j
					// 		// 		<< " | Iteration: " << num_client_cur_iters[j]
					// 		// 		<< " | Kernel Index: " << client_progress[j]
					// 		// 		<< " | Start Time: " << start_time_ns << " ns"
					// 		// 		<< std::endl;
					// 		schedule_kernel_profile(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
					// 		pop_from_queue(client_buffers[j], client_mutexes[j], j);
					// 		continue;
					// 	}
					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
					pop_from_queue(client_buffers[j], client_mutexes[j], j);
				}
			}
		}

		int finished = 0;
		for (int i=0; i<num_clients; i++) {
			
			if (
				(num_client_cur_iters[i] == num_client_max_iters[i])
				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
				|| (stop_ack[i] == true)
			)
				{
					finished += 1;
					// num_all_clients-=1;
					unsetmask(client_mutexes[i], i);
				}
			else if (seen[i] == num_client_kernels[i]) {
				// check if GPU work for this client has finished
				if (!locked[i]) {
					pthread_mutex_lock(client_mutexes[i]);
					locked[i] = true;
					DEBUG_PRINT("LOCK CLIENT %d\n", i);
				}
				bool ready = true;
				if (seq) {
					if (event_ids[0] >= 1) {
						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
							ready &= false;
					}
				}
				else {
					if (event_ids[i] >= 1) {
						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
							ready &= false;
					}
				}
				if (ready) {
					// unsetmask(client_mutexes[i], i);
					// num_all_clients +=1;
					// if yes, reset meta-structures for this client, and let it continue
					seen[i] = 0;
					if (seq)
						event_ids[0] = 0;
					event_ids[i] = 0;
					streams[i] = -1;
					fidx[i] = 0;
					request_status[i][num_client_cur_iters[i]] = true;
					//printf("UNLOCK CLIENT %d\n", i);
					pthread_mutex_unlock(client_mutexes[i]);
					num_client_cur_iters[i] += 1;
					locked[i] = false;
					client_progress[i] = 0;
					auto end = std::chrono::high_resolution_clock::now();
					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
					duration /= 1000.0;
					client_durations[i].push_back(duration);
				}
			}
		}

		if (finished==num_clients)
			break;
	}



	if (!warmup) {
		auto end_total = std::chrono::high_resolution_clock::now();
		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
		duration /= 1000.0;
		printf("Total loop took %f sec\n", duration);
		//process_eval(client_durations);
	}

	return NULL;
}
// // DP searching
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

	
	
// 	while(1) {
// 		int num_all_clients = num_clients;

// 		if (!is_train[hp_client]) {
// 			sm_threshold = max_sms;
// 			update_start = INT_MAX;
// 		}
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}
		
// 		int num_cur_client = 0;
// 		for (int j=0; j<num_clients; j++) {
// 			if (frecords[j] != NULL) { 
// 				unsetmask(client_mutexes[j], j);
// 				if ((frecords[j]->type != MALLOC_RECORD) && (frecords[j]->type != MEMCPY_RECORD) 
// 				&& (frecords[j]->type != MEMSET_RECORD) && (frecords[j]->type != FREE_RECORD)){
// 					num_cur_client++;
// 					continue;
// 				}
// 				schedule_kernel_KRISP_I(*(frecords[j]), 
// 				sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, client_mutexes[j]);
// 				pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 			}
// 		}
// 		// printf("size of current_schedule %ld\n",current_schedule.size());
// 		if(!current_schedule.empty() && num_tpcs ==24){
// 		// if(num_cur_client == num_all_clients){
// 		// for (int j=0; j<num_clients; j++) {
// 		// 	if (frecords[j] != NULL) { 
// 		// 		schedule_kernel_KRISP_I(*(frecords[j]), 
// 		// 		sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, client_mutexes[j]);
// 		// 		pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 		// 	}
// 		// }
// 			vector<std::pair<int, int>> client_to_schedule;
// 			bool found = false;
// 			for (size_t i = 0; i < current_schedule.size(); ++i) {
// 				if(found){
// 					break;
// 				}
// 				for(const auto& single_client_schedule: current_schedule[i]){
// 					if (single_client_schedule.first == hp_client) {
// 						client_to_schedule = current_schedule[i];
// 						current_schedule.erase(current_schedule.begin() + i);
// 						found = true;
// 						break;
// 					}
// 				}
// 			}
// 			if(!found){
// 				client_to_schedule = current_schedule.front();
// 				current_schedule.erase(current_schedule.begin());
// 			}

// 			// printf("client_to_schedule size %ld\n ", client_to_schedule.size());

// 			for(const auto& single_client_schedule: client_to_schedule){
// 				int j = single_client_schedule.first;
// 				printf("client %d, need %d tpcs\n", j, single_client_schedule.second);
// 				if(frecords[j] != NULL){
// 					setmask(client_mutexes[j], single_client_schedule.second, j);
// 					schedule_kernel_KRISP_I(*(frecords[j]), 
// 					sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, client_mutexes[j]);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 				else{
// 					printf("need abort\n");
// 					abort();
// 				}
// 			}
// 			// printf("after schedule, current_schedule size %ld\n ", current_schedule.size());
// 		}	
		
// 		else{

// 			if (num_cur_client == num_all_clients && num_tpcs == 24) {
// 				auto start = std::chrono::high_resolution_clock::now();
// 				int totalTPCs = 24;
// 				std::vector<std::vector<float>> latencyTable;

// 				for(int i = 0; i < num_cur_client; i++){
// 					op_info op_info = op_info_vector[i][seen[i]];
// 					latencyTable.push_back(op_info.profile_data);
// 				}

// 				std::vector<std::vector<std::vector<int>>> partitions;  // All partitions
// 				std::vector<std::vector<int>> partition;  // Current partition
// 				generate_partitions(num_cur_client, 0, partition, partitions);
// 				float min_tail_latency = INT_MAX;
// 				std::vector<std::vector<std::pair<int, int>>> best_allocation;
// 				for (const auto& partition : partitions) {
// 					auto [latency, allocation] = computeTailLatencyForPartition(partition, latencyTable, totalTPCs);
// 					if(latency < min_tail_latency){
// 						min_tail_latency = latency;
// 						best_allocation = allocation;
// 					}
// 					// std::cout << "Partition: ";
// 					// for (const auto& group : partition) {
// 					// 	std::cout << "{ ";
// 					// 	for (int client : group) {
// 					// 		std::cout << client << " ";
// 					// 	}
// 					// 	std::cout << "} ";
// 					// }
// 					// std::cout << " - Tail Latency: " << latency << std::endl;
// 				}
// 				for (size_t i = 0; i < best_allocation.size(); ++i) {
// 					std::cout << "Group " << i + 1 << ":" << std::endl; // Print the group number
// 					for (const auto& allocation : best_allocation[i]) {
// 						std::cout << "  Client " << allocation.first << " assigned " << allocation.second << " TPCs" << std::endl;
// 					}
// 					std::cout << std::endl; // Print a blank line between groups for readability
// 				}
// 				current_schedule = best_allocation;			
// 				// auto end = std::chrono::high_resolution_clock::now();
// 				// std::chrono::duration<double, std::milli> duration = end - start;
// 				// std::cout << "Seach Execution time: " << duration.count() << " ms" << std::endl;
// 			}
// 		}
// 		// Adjusting hp client
// 		int hp_iter = INT_MAX;
// 		int hp_progress = -1;

// 		for (int i = 0; i < num_clients; i++) {
// 			if(num_client_cur_iters[i] <= hp_iter){
// 				hp_iter = num_client_cur_iters[i];

// 			}
// 		}
// 		for (int i = 0; i < num_clients; i++) {
// 			if(num_client_cur_iters[i] == hp_iter){
// 				if(num_client_kernels[i] - client_progress[i] > hp_progress) {
// 					hp_progress = num_client_kernels[i] - client_progress[i];
// 					hp_client = i;
// 				}
// 			}
// 		}
// 			int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
			
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 					num_all_clients -=1;
// 					// printf("client %d finished !!!! num_all_clients: %d\n", i, num_all_clients);
// 					unsetmask(client_mutexes[i], i);
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// num_all_clients +=1;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					// printf("client %d is ready\n",i);
// 					unsetmask(client_mutexes[i], i);
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

//only scheule 2 kernels at a time
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

	
	
// 	while(1) {
// 		int num_all_clients = num_clients;

// 		if (!is_train[hp_client]) {
// 			sm_threshold = max_sms;
// 			update_start = INT_MAX;
// 		}
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}
		
// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
			
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 					num_all_clients -=1;
// 					// printf("client %d finished !!!! num_all_clients: %d\n", i, num_all_clients);
// 					unsetmask(client_mutexes[i], i);
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// num_all_clients +=1;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 		int num_cur_client = 0;
// 		for (int j=0; j<num_clients; j++) {
// 			if (frecords[j] != NULL) { 
// 				if ((frecords[j]->type != MALLOC_RECORD) && (frecords[j]->type != MEMCPY_RECORD) 
// 				&& (frecords[j]->type != MEMSET_RECORD) && (frecords[j]->type != FREE_RECORD)){
// 					num_cur_client++;
// 					continue;
// 				}
// 				schedule_kernel_KRISP_I(*(frecords[j]), 
// 				sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, client_mutexes[j]);
// 				pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 			}
// 		}
// 		vector<int> scheduled_clients;
// 		// printf("num_cur_client: %d, num_all_clients %d, num_tpcs: %d \n", num_cur_client, num_all_clients, num_tpcs);
// 		if (num_cur_client == num_all_clients && num_tpcs == 24) {
			
// 			scheduled_clients.push_back(hp_client);
// 			op_info op_info_hp_client = op_info_vector[hp_client][seen[hp_client]];

// 			float min_tail_latency = INT_MAX;
// 			for (int j = 0; j < num_clients; j++) {
// 				if (j != hp_client && frecords[j] != NULL) {
// 					op_info op_info = op_info_vector[j][seen[j]];
					
// 					for (int t_hp = 0; t_hp < num_tpcs - 1; t_hp++) { 
// 						int t_other = num_tpcs - (t_hp + 1); 
						
// 						float tail_latency = abs(op_info_hp_client.profile_data[t_hp] - op_info.profile_data[t_other]);

// 						if(tail_latency < min_tail_latency){
// 							min_tail_latency = tail_latency;
							
// 							client_mask[hp_client] = t_hp + 1;
// 							client_mask[j] = t_other;

// 							// printf("hp client kernel name: %s, selected tpcs: %d, it's performance %f\n",
// 							// op_info_hp_client.name.c_str(), t_hp + 1, op_info_hp_client.profile_data[t_hp]);

// 							// printf("normal client kernel name: %s, selected tpcs: %d, it's performance %f\n",
// 							// op_info.name.c_str(), t_other, op_info.profile_data[t_other]);

// 							// printf("tail latency is %f\n", tail_latency);

// 							scheduled_clients.clear();
// 							scheduled_clients.push_back(hp_client);
// 							scheduled_clients.push_back(j);
// 						}
// 					}
// 				}
// 			}
// 			for (int client : scheduled_clients) {
// 				if(frecords[client] != NULL){
// 					printf("schedule for client %d, his mask is %d\n", client, client_mask[client]);
// 					num_tpcs -= client_mask[client];
// 					setmask(client_mutexes[client], client_mask[client], client);
// 					schedule_kernel_KRISP_I(*(frecords[client]), 
// 											sched_streams[client], client, 
// 											events[client][event_ids[client]], 
// 											seen, event_ids, client, 
// 											client_mutexes[client]);
// 					pop_from_queue(client_buffers[client], client_mutexes[client], client);
// 				}
// 			}
// 			printf("-----------------------------\n");
// 		}

// 		int hp_iter = INT_MAX;
// 		int hp_progress = -1;

// 		for (int i = 0; i < num_clients; i++) {
// 			// printf("num_client_kernels: %d, client_progress: %d, num_client_cur_iters: %d\n",
// 			// num_client_kernels[i],client_progress[i],num_client_cur_iters[i]);

// 			if(num_client_cur_iters[i] <= hp_iter){
// 				hp_iter = num_client_cur_iters[i];

// 			}
// 		}
// 		for (int i = 0; i < num_clients; i++) {
			
// 			if(num_client_cur_iters[i] == hp_iter){
// 				if(num_client_kernels[i] - client_progress[i] > hp_progress) {
// 					hp_progress = num_client_kernels[i] - client_progress[i];
// 					hp_client = i;
// 				}
// 			}
// 		}
// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }
// temporal time sharing
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 0;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	while(1) {

// 		if (!is_train[hp_client]) {
// 			sm_threshold = max_sms;
// 			update_start = INT_MAX;
// 		}
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		if (frecords[hp_client] != NULL) { // high priority
// 			op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
// 			schedule_kernel_KRISP_I(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client, client_mutexes[hp_client]);
// 			streams[hp_client] = 1;
// 			profiles[hp_client] = op_info_1.profile;
// 			cur_sms[hp_client] = op_info_1.sm_used;
// 			status = 1;
// 			pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
// 			if ((frecords[hp_client]->type != MALLOC_RECORD) 
// 			&& (frecords[hp_client]->type != MEMCPY_RECORD) && (frecords[hp_client]->type != MEMSET_RECORD) 
// 			&& (frecords[hp_client]->type != FREE_RECORD)){
// 				client_progress[hp_client]+=1;
// 			}		
// 		}


// 		for (int t=0; t<num_clients; t++) {
// 			// if(t!=hp_client && t!=lp_client){
// 			if(t!=hp_client){
// 				// Do round-robin for the BE clients
// 				int j = t;
// 				if (frecords[j] != NULL) { // low priority
// 					op_info op_info_0 = op_info_vector[j][seen[j]];
// 					bool schedule = false;

// 					if ((num_clients==1) || (seen[hp_client]==0) || (frecords[j]->type == MALLOC_RECORD) || (frecords[j]->type == MEMCPY_RECORD) || (frecords[j]->type == MEMSET_RECORD) || (frecords[j]->type == FREE_RECORD))
// 						schedule = true;
// 					else if (num_client_cur_iters[j] < 10 || num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
// 						schedule = true;
// 					}
// 					else if (seen[hp_client] >= update_start && (op_info_0.sm_used <= sm_threshold && cudaEventQuery(*(events[hp_client][update_start-1])) == cudaSuccess)) // && (op_info_0.sm_used <= 10*sm_threshold))
// 						schedule = true;
// 					else if (seen[hp_client]>0 && (size + op_info_0.sm_used <= sm_threshold) &&  ((op_info_0.profile == -1 || profiles[hp_client]==-1 || (profiles[hp_client] != op_info_0.profile))))
// 						schedule = true;

// 					if (schedule) {

// 						size += op_info_0.sm_used;
// 						if ((frecords[j]->type != MALLOC_RECORD) && (frecords[j]->type != MEMCPY_RECORD) && (frecords[j]->type != MEMSET_RECORD) && (frecords[j]->type != FREE_RECORD))
// 							sum += op_info_0.duration;
// 						schedule_kernel_KRISP_I(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, client_mutexes[j]);
// 						status = 0;
// 						pop_from_queue(client_buffers[j], client_mutexes[j], j);

// 						streams[j] = 0;
// 						start = j;

// 						if ((frecords[j]->type != MALLOC_RECORD) 
// 						&& (frecords[j]->type != MEMCPY_RECORD) && (frecords[j]->type != MEMSET_RECORD) 
// 						&& (frecords[j]->type != FREE_RECORD)){
// 							client_progress[j]+=1;
// 						}	
// 					}
// 				}
// 			}
// 		}
		
		
// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
			
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 				finished += 1;
// 				unsetmask(client_mutexes[i], i);
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		int hp_iter = INT_MAX;
// 		int hp_progress = -1;

// 		for (int i = 0; i < num_clients; i++) {
// 			// printf("num_client_kernels: %d, client_progress: %d, num_client_cur_iters: %d\n",
// 			// num_client_kernels[i],client_progress[i],num_client_cur_iters[i]);

// 			if(num_client_cur_iters[i] <= hp_iter){
// 				hp_iter = num_client_cur_iters[i];

// 			}
// 		}
// 		for (int i = 0; i < num_clients; i++) {
			
// 			if(num_client_cur_iters[i] == hp_iter){
// 				if(num_client_kernels[i] - client_progress[i] > hp_progress) {
// 					hp_progress = num_client_kernels[i] - client_progress[i];
// 					hp_client = i;
// 				}
// 			}
// 		}


// 		if (finished==num_clients)
// 			break;
// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }



extern "C" {

	Scheduler* sched_init() {

		Scheduler* sched = new Scheduler();
		return sched;
	}


	void populate_kernel_info(char* kernel_info_file, vector<op_info> &ops) {

		// TODO: make this more generic, e.g. pass files/models w.r.t input
		printf("KERNEL_INFO_FILE IS %s\n", kernel_info_file);
		string line;
		std::ifstream infile(kernel_info_file);
		assert (infile.is_open());

		// ignore header
		std::getline(infile, line);

		while (std::getline(infile, line))
		{

			vector<string> v;
			stringstream sline = stringstream(line);
			while (sline.good()) {
        		string substr;
        		getline(sline, substr, ',');
        		v.push_back(substr);
    		}

			op_info info = {
				v[0],                               // Name
				stoi(v[1]),                         // Profile
				stoi(v[2]),                         // Memory footprint
				stoi(v[3]),                         // SM usage
				stof(v[4]),                         // Duration
				stoi(v[5]),                         // Grid
				stoi(v[6]),                         // Block
				stoi(v[7]),                     // Knee_TPC
				stoi(v[8]),                     // Is_Critical
				// vector<float>(), 				// Profile Data
        	};
			// for (size_t i = 8; i < v.size(); ++i) {
			// 	info.profile_data.push_back(stof(v[i])); 
			// }
			ops.push_back(info);
		}

		infile.close();

	}

	void setup_change(Scheduler* scheduler, int client_id, char* file, int num_kernels) {

		// needed for backward

		op_info_vector[client_id].clear();
		populate_kernel_info(file, op_info_vector[client_id]);
		int max_sm_used = 0;
		for (auto info: op_info_vector[client_id])
			max_sm_used = max(max_sm_used, info.sm_used);
		max_sms_clients[client_id] = max_sm_used;
		num_client_kernels[client_id] = num_kernels;

	}

	void setup(
		Scheduler* scheduler,
		int num_clients,
		int* tids,
		char** models,
		char** files,
		int* num_kernels,
		int* num_iters,
		bool* train,
		bool reef
	) {

		struct passwd *pw = getpwuid(getuid());
		char *homedir = pw->pw_dir;
		const char* lib_path = "/orion_bu/src/cuda_capture/libinttemp.so";

		klib = dlopen(strcat(homedir, lib_path), RTLD_NOW | RTLD_GLOBAL);

		if (!klib) {
			fprintf(stderr, "Error: %s\n", dlerror());
			return;
		}

#ifdef SYS_gettid
		pid_t mytid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif

		// 1. thread structures
		pid_t** thread_ids_all = (pid_t**)dlsym(klib, "thread_ids");
		*thread_ids_all = (pid_t*)malloc((2*num_clients+1)*sizeof(pid_t)); // 2*N threads + scheduler

		for (int i=0; i<num_clients; i++)
			(*thread_ids_all)[i] = tids[i];
		(*thread_ids_all)[num_clients] = mytid;
		for (int i=num_clients+1; i<2*num_clients+1; i++)
			(*thread_ids_all)[i] = 0;
		//printf("address is %p, %p\n", thread_ids_all, *thread_ids_all);

		int** num_total_clients = (int**)dlsym(klib, "num_total_clients");
		*num_total_clients = (int*)malloc(sizeof(int));
		**num_total_clients = num_clients;

		num_cur_clients.resize(num_clients);
		client_block.resize(num_clients);
		client_mask.resize(num_clients);
		is_executing.resize(num_clients);
		for (int i = 0; i < num_clients; ++i) {
			num_cur_clients[i] = i;
			client_block[i] = 0;
			client_mask[i] = 0;
			is_executing[i] = false;
		}
		client_finished = new bool[num_clients](); // Initialize all elements to false

		for (int i=0; i<=num_clients; i++) {
			DEBUG_PRINT("Scheduler setup the thread id at %d to be %d\n", i, (*thread_ids_all)[i]);
		}

		// 2. metadata structures
		for (int i=0; i<num_clients; i++) {
			op_info_vector.push_back({});
			client_durations.push_back({});
			populate_kernel_info(files[i], op_info_vector[i]);
			int max_sm_used = 0;
			for (auto info: op_info_vector[i])
				max_sm_used = max(max_sm_used, info.sm_used);
			max_sms_clients.push_back(max_sm_used);
			printf("----------- SIZE: %ld\n", op_info_vector[i].size());
			is_train.push_back(train[i]);
			client_progress.push_back(0);
			func_progress.push_back(-1);
		}

		// 3. indexes
		int** fidx_ptr = (int**)dlsym(klib, "func_indexes");
		*fidx_ptr = (int*)calloc(num_clients, sizeof(int));
		fidx = *fidx_ptr;

		num_client_kernels = num_kernels;
		num_client_max_iters = num_iters;

		num_client_cur_iters = (int*)calloc(num_clients, sizeof(int));
		locked = (bool*)calloc(num_clients, sizeof(bool));

		// to get measurements
		client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		total_client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		client_starts_set = (bool**)malloc(num_clients*sizeof(bool*));
		for (int i=0; i<num_clients; i++) {
			client_starts_set[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}

		// 4. communication queues + locks
		queue<func_record>*** buffers_ptr = (queue<func_record>***)dlsym(klib, "kqueues");
		*buffers_ptr = (queue<func_record>**)malloc(num_clients*sizeof(queue<func_record>*));
		queue<func_record>** buffers = *buffers_ptr;
		for (int i=0; i<num_clients; i++) {
			buffers[i] = new queue<func_record>();
			printf("buffer size is %ld\n", buffers[i]->size());
		}

		pthread_mutex_t*** client_mutexes_ptr = (pthread_mutex_t***)dlsym(klib, "mutexes");
		*client_mutexes_ptr = (pthread_mutex_t**)malloc(num_clients*sizeof(pthread_mutex_t*));
		client_mutexes = *client_mutexes_ptr;
		for (int i=0; i<num_clients; i++) {
			client_mutexes[i] = new pthread_mutex_t(); //(pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		}
		scheduler->profile_prep(buffers, num_clients, reef);

		// 5. runtime control
		bool*** request_status_ptr = (bool***)dlsym(klib, "client_request_status");
		*request_status_ptr = (bool**)malloc(num_clients*sizeof(bool*));
		request_status = *request_status_ptr;

		// check!
		bool** stops_ptr = (bool**)dlsym(klib, "client_stop");
		*stops_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stops = *stops_ptr;

		bool** stop_ack_ptr = (bool**)dlsym(klib, "client_stop_ack");
		*stop_ack_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stop_ack = *stop_ack_ptr;

		bool** affinity_set_ptr = (bool**)dlsym(klib, "affinity_set");
		(*affinity_set_ptr) = (bool*)calloc(num_clients+1, sizeof(bool));

		for (int i=0; i<num_clients; i++) {
			request_status[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}
	}


	void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int reef_depth, int hp_limit, int update_start) {
	// void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int reef_depth, int hp_limit, int update_start, int mymask) {

		printf("entered sched func!\n");
		if (profile_mode)
			scheduler->busy_wait_profile(num_clients, iter, warmup, warmup_iters, true, seq, reef_depth, hp_limit, update_start);
			// scheduler->busy_wait_profile(num_clients, iter, warmup, warmup_iters, true, seq, reef_depth, hp_limit, update_start, mymask);

		printf("exited sched func!\n");
		return NULL;
	}

	void* reset(Scheduler* scheduler, int num_clients) {
		scheduler->profile_reset(num_clients);
		return NULL;
	}
}