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

		// Priority Queue
		// set<int> scheduled_clients;	
		// set<int> unscheduled_clients;
		// while (!q_client.empty()) {
		// 	int client_idx = q_client.front();
		// 	q_client.pop();
		// 	int kernel_idx = seen[client_idx];
		// 	op_info op_info_cur = op_info_vector[client_idx][kernel_idx];
		// 	if (num_tpcs >= op_info_cur.Knee_TPC && frecords[client_idx] != NULL) {
		// 		scheduled_clients.insert(client_idx);
		// 		setmask(client_mutexes[client_idx], op_info_cur.Knee_TPC, client_idx);
		// 		client_block[client_idx] = 0;
		// 		int j = client_idx;
		// 		client_progress[client_idx] += 1;
		// 		schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
		// 		pop_from_queue(client_buffers[j], client_mutexes[j], j);
		// 	} else {
		// 		// printf("try to schedule for client %d but no success, cur number of tpcs: %d\n", client_idx, num_tpcs);
		// 		client_block[client_idx] +=1;
		// 		unscheduled_clients.insert(client_idx);
		// 	}
		// }

		// for (auto it = unscheduled_clients.rbegin(); it != unscheduled_clients.rend(); ++it) {
		// 	q_client.push(*it);
		// }


		for (int j=0; j<num_clients; j++) {
			// if(scheduled_clients.count(j) > 0 || unscheduled_clients.count(j) > 0){
			// 	continue;
			// }
			if (frecords[j] != NULL) {
				int kernel_idx = seen[j];
				op_info op_info_cur = op_info_vector[j][kernel_idx];
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
										// q_client.push(j);
										client_block[j] +=1;
										continue;  
									}
									tpc_usage = op_info_cur.Knee_TPC; 
								} else {
									tpc_usage = std::min(num_tpcs, op_info_cur.Knee_TPC);
								}

								setmask(client_mutexes[j], tpc_usage, j);
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