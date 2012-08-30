#include "CUDATimer.cuh"


CUDATimer::CUDATimer() {
    // TODO: some way of specifying EventCreate flags
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
}


CUDATimer::~CUDATimer() {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}


void CUDATimer::start(cudaStream_t stream) {
    cudaEventRecord(ev_start, stream);
}


float CUDATimer::get_elapsed_time_nosync(cudaStream_t stream) {
    float et;
    cudaEventRecord(ev_stop, stream);
    cudaEventElapsedTime(&et, ev_start, ev_stop);
    return et;
}

float CUDATimer::get_elapsed_time_sync(cudaStream_t stream) {
    float et;
    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&et, ev_start, ev_stop);
    return et;
}

