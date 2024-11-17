#include "utils_sched.h"
#include "libsmctrl.h"
#include <vector>
#include <algorithm>
#include <set>  
cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t (*malloc_function)(void** devPtr, size_t size);
cudaError_t (*free_function)(void* devPtr);
cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);

// cudnnStatus_t (*cudnn_create_function)(cudnnHandle_t *handle);
// cudnnStatus_t (*cudnn_backend_func)(cudnnHandle_t handle, const cudnnBackendDescriptor_t executionPlan, const cudnnBackendDescriptor_t varianPack);
// cudnnStatus_t (*cudnn_bnorm_reserve_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);
// cudnnStatus_t (*cudnn_conv_function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
// cudnnStatus_t (*cudnn_bnorm_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
// cudnnStatus_t (*cudnn_bnorm_infer_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);
// cudnnStatus_t (*cudnn_rnn_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);
// cudnnStatus_t (*cudnn_rnn_train_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);

// cudnnStatus_t (*cudnn_bnorm_bw_function)(
// 	cudnnHandle_t handle,
// 	cudnnBatchNormMode_t mode,
// 	cudnnBatchNormOps_t bnOps,
// 	const void *alphaDataDiff,
//     const void *betaDataDiff,
//     const void *alphaParamDiff,
//     const void *betaParamDiff,
//     const cudnnTensorDescriptor_t xDesc,
//     const void *xData,
//     const cudnnTensorDescriptor_t yDesc,
//     const void *yData,
//     const cudnnTensorDescriptor_t dyDesc,
//     const void *dyData,
//     const cudnnTensorDescriptor_t dzDesc,
//     void *dzData,
//     const cudnnTensorDescriptor_t dxDesc,
//     void *dxData,
//     const cudnnTensorDescriptor_t dBnScaleBiasDesc,
//     const void *bnScaleData,
//     const void *bnBiasData,
//     void *dBnScaleData,
//     void *dBnBiasData,
//     double epsilon,
//     const void *savedMean,
//     const void *savedInvVariance,
//     const cudnnActivationDescriptor_t activationDesc,
//     void *workspace,
//     size_t workSpaceSizeInBytes,
//     void *reserveSpace,
//     size_t reserveSpaceSizeInBytes
// );

// cudnnStatus_t (*cudnn_conv_bw_data_function)(
// 	cudnnHandle_t handle,
//     const void *alpha,
//     const cudnnFilterDescriptor_t wDesc,
//     const void *w,
//     const cudnnTensorDescriptor_t dyDesc,
//     const void *dy,
//     const cudnnConvolutionDescriptor_t convDesc,
//     cudnnConvolutionBwdDataAlgo_t algo,
//     void *workSpace,
//     size_t workSpaceSizeInBytes,
//     const void *beta,
//     const cudnnTensorDescriptor_t dxDesc,
//     void *dx
// );

// cudnnStatus_t (*cudnn_conv_bw_filter_function)(
// 	cudnnHandle_t handle,
//     const void *alpha,
//     const cudnnTensorDescriptor_t xDesc,
//     const void *x,
//     const cudnnTensorDescriptor_t dyDesc,
//     const void *dy,
//     const cudnnConvolutionDescriptor_t convDesc,
//     cudnnConvolutionBwdFilterAlgo_t algo,
//     void *workSpace,
//     size_t workSpaceSizeInBytes,
//     const void *beta,
//     const cudnnFilterDescriptor_t dwDesc,
//     void *dw
// );

cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);

// extern cudnnHandle_t* global_handle0;
// extern cudnnHandle_t* global_handle1;

extern int status;
extern int* seen;
extern int num_tpcs;
extern uint32_t mask;
extern uint32_t *localMask;
extern int* num_client_kernels;

int tpc_usage_count[24] = {0};


void process_eval(vector<vector<float>> &client_durations) {

	int num_clients = client_durations.size();
	for (int i=0; i<num_clients; i++) {
		vector<float> client_stats = client_durations[i];
		// remove first two iters
		client_stats.erase(client_stats.begin());
		client_stats.erase(client_stats.begin());

		sort(client_stats.begin(), client_stats.end());
		int client_len = client_stats.size();
		float p50 = client_stats[client_len/2];
		float p95 = client_stats[(client_len*95/100)];
		float p99 = client_stats[(client_len*99/100)];
		printf("Client %d, p50=%f, p95=%f, p99=%f\n", i, p50, p95, p99);
	}
}


void pop_from_queue(queue<struct func_record>* client_queue, pthread_mutex_t* client_mutex, int idx) {
	pthread_mutex_lock(client_mutex);
	client_queue->pop();
	pthread_mutex_unlock(client_mutex);
}

void create_streams(cudaStream_t** sched_streams, int num, bool reef) {

	int* lp = (int*)malloc(sizeof(int));
	int* hp = (int*)malloc(sizeof(int));

	CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(lp, hp));

	DEBUG_PRINT("Highest stream priority is %d, lowest stream priority is %d\n", *hp, *lp);
	assert(*lp==0);

	for (int i=0; i<num-1; i++) {
		sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, 0);
	}

	// client num-1 is high priority
	if (!reef) {
		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, *hp);
	}
	else {
		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, 0);
	}

}

void create_events(cudaEvent_t*** events, int num) {

	// per-stream event
	for (int i=0; i<num; i++) {
		events[i] = (cudaEvent_t**)malloc(30000*sizeof(cudaEvent_t*));
		for (int j=0; j<30000; j++) {
			//printf("create %d, %d\n", i, j);
			events[i][j] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
			CHECK_CUDA_ERROR(cudaEventCreateWithFlags(events[i][j], cudaEventDisableTiming));
		}
	}
}

void register_functions() {

    // for kernel
	*(void **)(&kernel_function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");
	assert(kernel_function != NULL);

	// for memcpy
	*(void **)(&memcpy_function) = dlsym (RTLD_DEFAULT, "cudaMemcpy");
	assert(memcpy_function != NULL);

	// for memcpy_async
	*(void **)(&memcpy_async_function) = dlsym (RTLD_DEFAULT, "cudaMemcpyAsync");
	assert(memcpy_async_function != NULL);

	// for malloc
	*(void **)(&malloc_function) = dlsym (RTLD_DEFAULT, "cudaMalloc");
	assert(malloc_function != NULL);

	// for free
	*(void **)(&free_function) = dlsym (RTLD_DEFAULT, "cudaFree");
	assert(free_function != NULL);

	// for memset
	*(void **)(&memset_function) = dlsym (RTLD_DEFAULT, "cudaMemset");
	assert (memset_function != NULL);

	// for memset_async
	*(void **)(&memset_async_function) = dlsym (RTLD_DEFAULT, "cudaMemsetAsync");
	assert (memset_async_function != NULL);

	// // for cudnn create
	// *(void **)(&cudnn_create_function) = dlsym(RTLD_DEFAULT, "cudnnCreate");
	// assert(cudnn_create_function != NULL);

	// // for cudnn_bnorm_reserve
	// *(void **)(&cudnn_bnorm_reserve_function) = dlsym(RTLD_DEFAULT, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
	// assert(cudnn_bnorm_reserve_function != NULL);

	// // for cudnn conv
	// *(void **)(&cudnn_backend_func) = dlsym(RTLD_DEFAULT, "cudnnBackendExecute");
	// assert(cudnn_backend_func != NULL);

	// // for cudnn conv
	// *(void **)(&cudnn_conv_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionForward");
	// assert(cudnn_conv_function != NULL);

	// // for bnorm train
	// *(void **)(&cudnn_bnorm_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardTrainingEx");
	// assert(cudnn_bnorm_function != NULL);

	// // for bnorm infer
	// *(void **)(&cudnn_bnorm_infer_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardInference");
	// assert(cudnn_bnorm_infer_function != NULL);

	// // for rnn infer
	// *(void **)(&cudnn_rnn_function) = dlsym(RTLD_DEFAULT, "cudnnRNNForwardInference");
	// assert(cudnn_rnn_function != NULL);

	// // for rnn train
	// *(void **)(&cudnn_rnn_train_function) = dlsym(RTLD_DEFAULT, "cudnnRNNForwardTraining");
	// assert(cudnn_rnn_train_function != NULL);

	// // for bnorm backward
	// *(void **)(&cudnn_bnorm_bw_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationBackwardEx");
	// assert(cudnn_bnorm_bw_function != NULL);

	// // for conv data backward
	// *(void **)(&cudnn_conv_bw_data_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionBackwardData");
	// assert(cudnn_conv_bw_data_function != NULL);

	// // for conv filter backward
	// *(void **)(&cudnn_conv_bw_filter_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionBackwardFilter");
	// assert(cudnn_conv_bw_filter_function != NULL);

	// CUBLAS sgemm
	*(void **)(&cublas_sgemm_function) = dlsym(RTLD_DEFAULT, "cublasSgemm_v2");
	assert(cublas_sgemm_function != NULL);

	// CUBLAS sgemm strided
	*(void **)(&cublas_sgemm_strided_function) = dlsym(RTLD_DEFAULT, "cublasSgemmStridedBatched");
	assert(&cublas_sgemm_strided_function != NULL);

}

void schedule_kernel_profile(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid) {
	
	switch (frecord.type) {
		case KERNEL_RECORD: {
			DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			cudaEvent_t start, stop;
			cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
			cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
			cudaEventRecord(start, *sched_stream);

			(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			cudaEventRecord(stop, *sched_stream);
			cudaEventSynchronize(stop);
			// cudaDeviceSynchronize();
			// cudaStreamWaitEvent(*sched_stream, stop, 0);  
			// cudaStreamSynchronize(*sched_stream);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			long long nanoseconds = static_cast<long long>(milliseconds * 1e6);
			printf("Kernel execution time: %lld ns\n", nanoseconds);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);

			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			cudaEvent_t start, stop;
			cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
			cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
			cudaEventRecord(start, *sched_stream);
			(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			cudaEventRecord(stop, *sched_stream);
			cudaEventSynchronize(stop); 
			// cudaDeviceSynchronize();
			// cudaStreamWaitEvent(*sched_stream, stop, 0);  
			// cudaStreamSynchronize(*sched_stream);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			long long nanoseconds = static_cast<long long>(milliseconds * 1e6);
			printf("Kernel Sgemm execution time: %lld ns\n", nanoseconds);

			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);

			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();
			// abort(1, 0, "UNSUPPORTED OPERATION - ABORT\n");

	}
	DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}
void schedule_kernel(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid) {

	switch (frecord.type) {
		case KERNEL_RECORD: {
			DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);

			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);

			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();

	}
	DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}



void schedule_kernel_KRISP(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid, vector<int> num_clients) {
	libsmctrl_set_mask();
	switch (frecord.type) {
		case KERNEL_RECORD: {
			DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			// printf("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			// printf("found a new MEMCPY_RECORD record from idx %d!\n", idx);
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			// printf("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			// printf("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				// printf("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				// printf("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);
			// printf("found a new cublas sgemm record from idx %d!\n", idx);
			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);
			// printf("found a new cublas sgemm strided record from idx %d!\n", idx);
			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();
			// abort(1, 0, "UNSUPPORTED OPERATION - ABORT\n");

	}
	DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}

void setmask(pthread_mutex_t* client_mutex, int num_tpcs_need, int idx) {
    pthread_mutex_lock(client_mutex);

    // Print the mask before setting
    // std::cout << "Mask before setting: " << std::bitset<32>(mask) << std::endl;
    // std::cout << "localMask[" << idx << "] before setting: " << std::bitset<32>(localMask[idx]) << std::endl;

    int count = 0;
    for (int i = 0; i < 24 && count < num_tpcs_need; ++i) {
        if ((mask & (1 << i)) == 0) { // Check if the bit is 0
            mask |= (1 << i); // Set the bit to 1
            localMask[idx] |= (1 << i);
            ++count;
        }
    }

    if (count < num_tpcs_need) {
        std::cerr << "Error: Insufficient 0 bits available in the mask to set " 
                  << num_tpcs_need << " bits." << std::endl;
        abort();
    }


    num_tpcs -= count;
    uint32_t mask_to_set = ~localMask[idx];
	// std::cout << "Mask after setting: " << std::bitset<32>(mask) << std::endl;
    // std::cout << "localMask[" << idx << "] after setting: " << std::bitset<32>(localMask[idx]) << std::endl;
    libsmctrl_set_KRISP_I_mask(mask_to_set);
    pthread_mutex_unlock(client_mutex);
	// abort();
}

void setmask_O(pthread_mutex_t* client_mutex, int num_tpcs_need, int idx){
	pthread_mutex_lock(client_mutex);
    int count = 0;

	int tpc_indices[24];
    for (int i = 0; i < 24; ++i) {
        tpc_indices[i] = i;
    }

	std::sort(tpc_indices, tpc_indices + 24, [](int a, int b) {
        return tpc_usage_count[a] < tpc_usage_count[b];
    });

    for (int i = 0; i < 24 && count < num_tpcs_need; ++i) {
        int tpc = tpc_indices[i];
		localMask[idx] |= (1 << tpc); 
		tpc_usage_count[tpc]++;
		++count;
    }

	uint32_t mask_to_set = ~localMask[idx];
	libsmctrl_set_KRISP_I_mask(mask_to_set);
	pthread_mutex_unlock(client_mutex);
}


void unsetmask(pthread_mutex_t* client_mutex, int idx){
	pthread_mutex_lock(client_mutex);
	int count = 0;
    for (int i = 0; i < 24; ++i) {
        // Check if the bit in the local mask is 1
        if ((localMask[idx] & (1 << i)) != 0) {
            // If the corresponding bit in the original mask is 1, set it to 0
            if ((mask & (1 << i)) != 0) {
                mask &= ~(1 << i); // Set the bit to 0
				count++;
            }
			else{
				printf("error setting the mask\n");
				abort();
			}
        }
    }
	// std::cout << "mask after unset" << std::bitset<32>(mask) <<" from idx: "<<idx<< std::endl;
	localMask[idx] = 0;
	num_tpcs += count;
	pthread_mutex_unlock(client_mutex);
}
void unsetmask_nomutex(int idx){
	int count = 0;
    for (int i = 0; i < 24; ++i) {
        // Check if the bit in the local mask is 1
        if ((localMask[idx] & (1 << i)) != 0) {
            // If the corresponding bit in the original mask is 1, set it to 0
            if ((mask & (1 << i)) != 0) {
                mask &= ~(1 << i); // Set the bit to 0
				count++;
            }
			else{
				printf("error setting the mask\n");
				abort();
			}
        }
    }
	// std::cout << "mask after unset" << std::bitset<32>(mask) <<" from idx: "<<idx<< std::endl;
	localMask[idx] = 0;
	num_tpcs += count;
}

void unsetmask_O(pthread_mutex_t* client_mutex, int idx) {
    pthread_mutex_lock(client_mutex);

    for (int i = 0; i < 24; ++i) {
        if (localMask[idx] & (1 << i)) { // If the TPC was used in this kernel.
            tpc_usage_count[i]--; // Decrement the usage count.
            localMask[idx] &= ~(1 << i); // Clear the bit in the local mask for this TPC.
        }
    }
	
    pthread_mutex_unlock(client_mutex);
}

void schedule_kernel_KRISP_I(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid, pthread_mutex_t* client_mutex) {
	switch (frecord.type) {
		case KERNEL_RECORD: {
			DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			
			// cudaEvent_t start, stop;
			// cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
			// cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
			// cudaEventRecord(start, *sched_stream);

			(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			// cudaEventRecord(stop, *sched_stream);
			// cudaEventSynchronize(stop); // Ensure the event has been recorded
			// cudaDeviceSynchronize();

			// float milliseconds = 0;
			// cudaEventElapsedTime(&milliseconds, start, stop);
			// long long nanoseconds = static_cast<long long>(milliseconds * 1e6);
			// printf("Kernel execution time: %lld ns\n", nanoseconds);
			// cudaEventDestroy(start);
			// cudaEventDestroy(stop);

			CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);
			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);


			// cudaEvent_t start, stop;
			// cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
			// cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
			// cudaEventRecord(start, *sched_stream);
			(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			// cudaEventRecord(stop, *sched_stream);
			// cudaEventSynchronize(stop); // Ensure the event has been recorded
			// cudaDeviceSynchronize();
			// float milliseconds = 0;
			// cudaEventElapsedTime(&milliseconds, start, stop);
			// long long nanoseconds = static_cast<long long>(milliseconds * 1e6);
			// printf("Kernel Sgemm execution time: %lld ns\n", nanoseconds);

			// cudaEventDestroy(start);
			// cudaEventDestroy(stop);

			cublasSetStream_v2(record.handle, 0);
			CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);
			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();
			// abort(1, 0, "UNSUPPORTED OPERATION - ABORT\n");

	}
	// unsetmask(client_mutex, idx);
	DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}