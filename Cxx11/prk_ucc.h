#ifndef PRK_UCC_H
#define PRK_UCC_H

#include <ucc/api/ucc.h>
#include <cstdlib>
#include <iostream>

static inline void prk_ucc_check(ucc_status_t status, const char* file, int line) {
    if (status != UCC_OK) {
        std::cerr << "UCC error at " << file << ":" << line 
                  << " code: " << status << std::endl;
        std::exit(1);
    }
}

#define PRK_UCC_CHECK(stmt) prk_ucc_check(stmt, __FILE__, __LINE__)

class prk_ucc {
    ucc_lib_h lib;
    ucc_context_h context;
    ucc_team_h team;
    int rank;
    int nranks;

public:
    prk_ucc() {
        // Initialize UCC library
        ucc_lib_config_h lib_config;
        ucc_lib_params_t lib_params = {0};
        
        PRK_UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
        PRK_UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
        ucc_lib_config_release(lib_config);

        // Create context
        ucc_context_params_t ctx_params = {0};
        PRK_UCC_CHECK(ucc_context_create(lib, &ctx_params, &context));

        // Create team
        ucc_team_params_t team_params = {0};
        team_params.oob.allgather = NULL;  // Set appropriate OOB
        team_params.oob.req_test = NULL;   // Set appropriate OOB
        team_params.oob.req_free = NULL;   // Set appropriate OOB
        team_params.oob.n_oob_eps = 0;     // Set appropriate value
        team_params.oob.comm_size = 0;     // Set from external info
        
        PRK_UCC_CHECK(ucc_team_create_post(&team_params, &team));
        
        ucc_status_t status;
        do {
            status = ucc_team_create_test(team);
        } while (status == UCC_INPROGRESS);
        PRK_UCC_CHECK(status);

        // Get rank info
        PRK_UCC_CHECK(ucc_team_get_rank(team, &rank));
        PRK_UCC_CHECK(ucc_team_size(team, &nranks));
    }

    ~prk_ucc() {
        if (team) {
            ucc_team_destroy(team);
        }
        if (context) {
            ucc_context_destroy(context);
        }
        if (lib) {
            ucc_finalize(lib);
        }
    }

    int get_rank() const { return rank; }
    int get_size() const { return nranks; }
    ucc_team_h get_team() const { return team; }

    void barrier() {
        ucc_coll_args_t coll_args = {0};
        coll_args.mask = 0;
        coll_args.coll_type = UCC_COLL_TYPE_BARRIER;

        ucc_coll_req_h request;
        PRK_UCC_CHECK(ucc_collective_init(&coll_args, &request, team));
        PRK_UCC_CHECK(ucc_collective_post(request));
        
        ucc_status_t status;
        do {
            status = ucc_collective_test(request);
        } while (status == UCC_INPROGRESS);
        PRK_UCC_CHECK(status);
        
        ucc_collective_finalize(request);
    }

    void alltoall(void* sendbuf, void* recvbuf, size_t count, ucc_datatype_t dtype) {
        ucc_coll_args_t coll_args = {0};
        coll_args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.coll_type = UCC_COLL_TYPE_ALLTOALL;
        coll_args.src.info.buffer = sendbuf;
        coll_args.src.info.count = count;
        coll_args.src.info.datatype = dtype;
        coll_args.dst.info.buffer = recvbuf;
        coll_args.dst.info.count = count;
        coll_args.dst.info.datatype = dtype;

        ucc_coll_req_h request;
        PRK_UCC_CHECK(ucc_collective_init(&coll_args, &request, team));
        PRK_UCC_CHECK(ucc_collective_post(request));
        
        ucc_status_t status;
        do {
            status = ucc_collective_test(request);
        } while (status == UCC_INPROGRESS);
        PRK_UCC_CHECK(status);
        
        ucc_collective_finalize(request);
    }
};

#endif // PRK_UCC_H
