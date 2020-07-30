#ifndef PRK_CUDA_HPP
#define PRK_CUDA_HPP

//#include <cstdio>
//#include <cstdlib>

#include <iostream>
#include <vector>
#include <array>

#ifndef __NVCC__
#warning Please compile CUDA code with CC=nvcc.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#endif

#if defined(PRK_USE_CUBLAS)
#if defined(__NVCC__)
#include <cublas_v2.h>
#else
#error Sorry, no CUBLAS without NVCC.
#endif
#endif

#ifdef __CORIANDERCC__
// Coriander does not support double
typedef float prk_float;
#else
typedef double prk_float;
#endif

namespace prk
{
    namespace CUDA
    {
        void check(cudaError_t rc)
        {
            if (rc==cudaSuccess) {
                return;
            } else {
                std::cerr << "PRK CUDA error: " << cudaGetErrorString(rc) << std::endl;
                std::abort();
            }
        }

#if defined(PRK_USE_CUBLAS)
        // It seems that Coriander defines cublasStatus_t to cudaError_t
        // because the compiler complains that this is a redefinition.
        void check(cublasStatus_t rc)
        {
            if (rc==CUBLAS_STATUS_SUCCESS) {
                return;
            } else {
                std::cerr << "PRK CUBLAS error: " << rc << std::endl;
                std::abort();
            }
        }
#endif

        class info {

            private:
                int nDevices;
                std::vector<cudaDeviceProp> vDevices;

            public:
                int maxThreadsPerBlock;
                std::array<unsigned,3> maxThreadsDim;
                std::array<unsigned,3> maxGridSize;

                info() {
                    prk::CUDA::check( cudaGetDeviceCount(&nDevices) );
                    vDevices.resize(nDevices);
                    for (int i=0; i<nDevices; ++i) {
                        cudaGetDeviceProperties(&(vDevices[i]), i);
                        if (i==0) {
                            maxThreadsPerBlock = vDevices[i].maxThreadsPerBlock;
                            for (int j=0; j<3; ++j) {
                                maxThreadsDim[j]   = vDevices[i].maxThreadsDim[j];
                                maxGridSize[j]     = vDevices[i].maxGridSize[j];
                            }
                        }
                    }
                }

                // do not use cached value as a hedge against weird stuff happening
                int num_gpus() {
                    int g;
                    prk::CUDA::check( cudaGetDeviceCount(&g) );
                    return g;
                }

                int get_gpu() {
                    int g;
                    prk::CUDA::check( cudaGetDevice(&g) );
                    return g;
                }

                void set_gpu(int g) {
                    prk::CUDA::check( cudaSetDevice(g) );
                }

                void print() {
                    for (int i=0; i<nDevices; ++i) {
                        std::cout << "device name: " << vDevices[i].name << "\n";
#ifndef __CORIANDERCC__
                        std::cout << "total global memory:     " << vDevices[i].totalGlobalMem << "\n";
                        std::cout << "max threads per block:   " << vDevices[i].maxThreadsPerBlock << "\n";
                        std::cout << "max threads dim:         " << vDevices[i].maxThreadsDim[0] << ","
                                                                 << vDevices[i].maxThreadsDim[1] << ","
                                                                 << vDevices[i].maxThreadsDim[2] << "\n";
                        std::cout << "max grid size:           " << vDevices[i].maxGridSize[0] << ","
                                                                 << vDevices[i].maxGridSize[1] << ","
                                                                 << vDevices[i].maxGridSize[2] << "\n";
                        std::cout << "memory clock rate (KHz): " << vDevices[i].memoryClockRate << "\n";
                        std::cout << "memory bus width (bits): " << vDevices[i].memoryBusWidth << "\n";
#endif
                    }
                }

                bool checkDims(dim3 dimBlock, dim3 dimGrid) {
                    if (dimBlock.x > maxThreadsDim[0]) {
                        std::cout << "dimBlock.x too large" << std::endl;
                        return false;
                    }
                    if (dimBlock.y > maxThreadsDim[1]) {
                        std::cout << "dimBlock.y too large" << std::endl;
                        return false;
                    }
                    if (dimBlock.z > maxThreadsDim[2]) {
                        std::cout << "dimBlock.z too large" << std::endl;
                        return false;
                    }
                    if (dimGrid.x  > maxGridSize[0])   {
                        std::cout << "dimGrid.x  too large" << std::endl;
                        return false;
                    }
                    if (dimGrid.y  > maxGridSize[1]) {
                        std::cout << "dimGrid.y  too large" << std::endl;
                        return false;
                    }
                    if (dimGrid.z  > maxGridSize[2]) {
                        std::cout << "dimGrid.z  too large" << std::endl;
                        return false;
                    }
                    return true;
                }
        };

        template<typename T>
        T * alloc(size_t n, int device = 0) {
            size_t bytes = n * sizeof(T);
            T * ptr = nullptr;
            //std::cerr << "INFO: alloc - device: " << device << std::endl;
            if (device < 0) {
                prk::CUDA::check( cudaMallocHost((void**)&ptr, bytes) );
            } else {
                prk::CUDA::check( cudaSetDevice(device) );
                prk::CUDA::check( cudaMalloc((void**)&ptr, bytes) );
            }
            return ptr;
        }

        template<typename T>
        void free(T * ptr, int device = 0) {
            //std::cerr << "INFO: free - device: " << device << std::endl;
            if (device < 0) {
                prk::CUDA::check( cudaFreeHost(ptr) );
            } else {
                prk::CUDA::check( cudaSetDevice(device) );
                prk::CUDA::check( cudaFree(ptr) );
            }
        }

        // This is copied from prk_util.h and only changes the allocation,
        // which is pathetic and should be improved later.
        template <typename T>
        class vector {

            private:
                T * data_;
                size_t size_;

            public:

                vector(size_t n) {
                    this->data_ = prk::CUDA::alloc<T>(n,-1);
                    this->size_ = n;
                }

                vector(size_t n, T v) {
                    this->data_ = prk::CUDA::alloc<T>(n,-1);
                    for (size_t i=0; i<n; ++i) this->data_[i] = v;
                    this->size_ = n;
                }

                ~vector() {
                    prk::CUDA::free<T>(this->data_,-1);
                }

                void operator~() {
                    this->~vector();
                }

                T * data() {
                    return this->data_;
                }

                size_t size() {
                    return this->size_;
                }

                T const & operator[] (size_t n) const {
                    return this->data_[n];
                }

                T & operator[] (size_t n) {
                    return this->data_[n];
                }

                T * begin() {
                    return &(this->data_[0]);
                }

                T * end() {
                    return &(this->data_[this->size_]);
                }

                void fill(T v) {
                    for (size_t i=0; i<this->size_; ++i) this->data_[i] = v;
                }
        };

        class queues {

            private:
                int np_;

            public:
                queues(void)
                {
                    prk::CUDA::check( cudaGetDeviceCount(&np_) );
                    //std::cerr << "INFO: device count: " << np_ << std::endl;
                }

                queues(int n)
                {
                    int d;
                    prk::CUDA::check( cudaGetDeviceCount(&d) );
                    if (n > d) {
                        std::cerr << "ERROR: requesting more devices (" << n << ") than available (" << d << ")" << std::endl;
                        std::abort();
                    }
                    //std::cerr << "INFO: device count: " << np_ << std::endl;
                }

                void resize(int n)
                {
                    int d;
                    prk::CUDA::check( cudaGetDeviceCount(&d) );
                    if (n > d) {
                        std::cerr << "ERROR: requesting more devices (" << n << ") than available (" << d << ")" << std::endl;
                        std::abort();
                    }
                    np_ = n;
                    //std::cerr << "INFO: device count: " << np_ << std::endl;
                }

                int size(void)
                {
                    return np_;
                }

                void wait(int i)
                {
                    if (i > np_) {
                        std::cerr << "ERROR: invalid device id: " << i << std::endl;
                        std::abort();
                    }
                    prk::CUDA::check( cudaSetDevice(i) );
                    prk::CUDA::check( cudaDeviceSynchronize() );
                }

                void waitall(void)
                {
                    for (int i=0; i<np_; ++i) {
                        prk::CUDA::check( cudaSetDevice(i) );
                        prk::CUDA::check( cudaDeviceSynchronize() );
                    }
                }

                template <typename T>
                void allocate(std::vector<T*> & device_pointers,
                              size_t num_elements)
                {
                    //std::cerr << "INFO: device count: " << np_ << std::endl;
                    //std::cerr << "INFO: device_pointers.size(): " << device_pointers.size() << std::endl;
                    int np = device_pointers.size();
                    for (int i=0; i<np; ++i) {
                        device_pointers.at(i) = prk::CUDA::alloc<T>(num_elements, i);
                        //std::cerr << "INFO: allocate - device, address: " << i << ", " << device_pointers.at(i) << std::endl;
                    }
                }

                template <typename T>
                void free(std::vector<T*> & device_pointers)
                {
                    //std::cerr << "INFO: device count: " << np_ << std::endl;
                    //std::cerr << "INFO: device_pointers.size(): " << device_pointers.size() << std::endl;
                    int np = device_pointers.size();
                    for (int i=0; i<np; ++i) {
                        //std::cerr << "INFO: free - device, address: " << i << ", " << device_pointers.at(i) << std::endl;
                        prk::CUDA::free(device_pointers.at(i), i);
                    }
                }

                template <typename T, typename B>
                void broadcast(std::vector<T*> & device_pointers,
                               const B & host_pointer,
                               size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    int np = device_pointers.size();
                    for (int i=0; i<np; ++i) {
                        auto target = device_pointers.at(i);
                        auto source = &host_pointer[0];
                        std::cout << "BCAST: device " << i << std::endl;
                        prk::CUDA::check( cudaSetDevice(i) );
                        prk::CUDA::check( cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice) );
                    }
                }

                template <typename T, typename B>
                void reduce(B & host_pointer,
                            const std::vector<T*> & device_pointers,
                            size_t num_elements)
                {
                    std::cout << "REDUCE: num_elements " << num_elements << std::endl;
                    auto bytes = num_elements * sizeof(T);
                    std::cout << "REDUCE: bytes " << bytes << std::endl;
                    auto temp = prk::CUDA::vector<T>(num_elements, 0);
                    int np = device_pointers.size();
                    for (int i=0; i<np; ++i) {
                        std::cout << "REDUCE: device " << i << std::endl;
                        auto target = &temp[0];
                        auto source = device_pointers.at(i);
                        prk::CUDA::check( cudaSetDevice(i) );
                        prk::CUDA::check( cudaMemcpy(target, source, bytes, cudaMemcpyDeviceToHost) );
                        for (size_t e=0; e<num_elements; ++e) {
                            host_pointer[e] += temp[e];
                        }
                    }
                }

                template <typename T, typename B>
                void gather(B & host_pointer,
                            const std::vector<T*> & device_pointers,
                            size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    int np = device_pointers.size();
                    for (int i=0; i<np; ++i) {
                        auto target = &host_pointer[i * num_elements];
                        auto source = device_pointers.at(i);
                        prk::CUDA::check( cudaSetDevice(i) );
                        prk::CUDA::check( cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost) );
                    }
                }

                template <typename T, typename B>
                void scatter(std::vector<T*> & device_pointers,
                             const B & host_pointer,
                             size_t num_elements)
                {
                    auto bytes = num_elements * sizeof(T);
                    int np = device_pointers.size();
                    for (int i=0; i<np; ++i) {
                        auto target = device_pointers.at(i);
                        auto source = &host_pointer[i * num_elements];
                        prk::CUDA::check( cudaSetDevice(i) );
                        prk::CUDA::check( cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice) );
                    }
                }

#if 0
                // num_elements is defined the same as MPI
                // each device contributes np * num_elements
                // each device receives np * num_elements
                template <typename T>
                void alltoall(std::vector<T*> & device_pointers_out,
                              std::vector<T*> & device_pointers_in,
                              size_t num_elements)
                {
                    if ( device_pointers_out.size() != device_pointers_in.size() ) {
                        std::cerr << "ERROR: device pointer vectors do not match: "
                                  << device_pointers_out.size() << "!=" << device_pointers_in.size() << std::endl;
                        std::abort();
                    }
                    auto bytes = num_elements * sizeof(T);
                    // allocate np*np temp space on the host, because
                    // we cannot copy device-to-device if they are in
                    // different contexts.
                    // we can specialize for single-context later...
                    int np = device_pointers.size();
                    prk::vector<double> temp(num_elements * np * np);

                    // gather phase - contiguous
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = &temp[i * np * num_elements];
                        auto source = device_pointers_in.at(i);
                        v.memcpy(target, source, np * bytes);
                    }

                    // scatter phase - noncontiguous
                    for (const auto & l : list | boost::adaptors::indexed(0) ) {
                        auto i = l.index();
                        auto v = l.value();
                        auto target = device_pointers_out.at(i);
                        auto source = &temp[i * num_elements];
                        v.memcpy(target, source, bytes);
                    }

                }
#endif
        };

    } // CUDA namespace

} // prk namespace

#endif // PRK_CUDA_HPP
