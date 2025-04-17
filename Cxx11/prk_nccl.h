#ifndef PRK_NCCL_HPP
#define PRK_NCCL_HPP

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <typeinfo>
#include <map>

#include <nccl.h>

#include "prk_cuda.h"

namespace prk
{
    void check(ncclResult_t rc)
    {
        if (rc != ncclSuccess) {
            std::cerr << "PRK NCCL error: " << ncclGetErrorString(rc) << std::endl;
            std::abort();
        }
    }

    namespace NCCL
    {
        ncclComm_t nccl_comm_world;

        std::map<void*,void*> address_handle_map{};

        ncclComm_t get_world(void)
        {
            return nccl_comm_world;
        }

        void init(int np, ncclUniqueId uniqueId, int me)
        {
            prk::check( ncclGroupStart() );
            prk::check( ncclCommInitRank(&nccl_comm_world, np, uniqueId, me) );
            prk::check( ncclGroupEnd() );
        }

        void finalize(void)
        {
            prk::check( ncclCommDestroy(nccl_comm_world) );
        }

        template <typename T>
        T * alloc(size_t n, bool register_ = true)
        {
            T * ptr;
            size_t bytes = n * sizeof(T);
            prk::check( ncclMemAlloc((void**)&ptr, bytes) );
            if (register_) {
                void * handle;
                prk::check( ncclCommRegister(nccl_comm_world, (void**)&ptr, bytes, &handle) );
                address_handle_map.insert_or_assign(ptr, handle);
            }
            return ptr;
        }

        template <typename T>
        void free(T * ptr)
        {
            try {
                void * handle = address_handle_map.at(ptr);
                prk::check( ncclCommDeregister(nccl_comm_world, handle) );
                address_handle_map.erase(ptr);
            }
            catch(const std::out_of_range& ex)
            {
                // no problem - the memory was not registered
            }
            prk::check( ncclMemFree((void*)ptr) );
        }
        
        template <typename T>
        ncclDataType_t get_NCCL_Datatype(T t) { 
            std::cerr << "get_NCCL_Datatype resolution failed for type " << typeid(T).name() << std::endl;
            std::abort();
        }

        template <>
        constexpr ncclDataType_t get_NCCL_Datatype(double d) { return ncclFloat64; }
        template <>
        constexpr ncclDataType_t get_NCCL_Datatype(int i) { return ncclInt32; }

        template <typename T>
        void alltoall(const T * sbuffer, T * rbuffer, size_t count, cudaStream_t stream = 0) {
            ncclDataType_t type = get_NCCL_Datatype(*sbuffer);

            int np;
            prk::check( ncclCommCount(nccl_comm_world, &np) );

            prk::check( ncclGroupStart() );
            for (int r=0; r<np; r++) {
                prk::check( ncclSend(sbuffer + r*count, count, type, r, nccl_comm_world, stream) );
                prk::check( ncclRecv(rbuffer + r*count, count, type, r, nccl_comm_world, stream) );
            }
            prk::check( ncclGroupEnd() );
        }

        template <typename T>
        void sendrecv(const T * sbuffer,  int dst, T * rbuffer, int src, size_t count, cudaStream_t stream = 0) {
            ncclDataType_t type = get_NCCL_Datatype(*sbuffer);
            prk::check( ncclGroupStart() );
            {
                prk::check( ncclSend(sbuffer, count, type, dst, nccl_comm_world, stream) );
                prk::check( ncclRecv(rbuffer, count, type, src, nccl_comm_world, stream) );
            }
            prk::check( ncclGroupEnd() );
        }

    } // NCCL namespace

} // prk namespace

#endif // PRK_NCCL_HPP
