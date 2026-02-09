#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <iostream>
#include <cuda_runtime_api.h>

// Macro to check CUDA API calls (malloc, memcpy, etc.)
#define CHECK_CUDA_CALL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA API Error: " #call << std::endl; \
            std::cerr << "  " << cudaGetErrorString(err) << " (code: " << err << ")" << std::endl; \
            std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

// Macro to check CUDA errors after kernel launches
#define CHECK_CUDA_ERROR(kernel_name) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error after " << kernel_name << ": " \
                      << cudaGetErrorString(err) << " (code: " << err << ")" << std::endl; \
            std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error (sync) after " << kernel_name << ": " \
                      << cudaGetErrorString(err) << " (code: " << err << ")" << std::endl; \
            std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

#endif // CUDA_ERROR_CHECK_H
