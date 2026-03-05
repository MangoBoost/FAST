#include <iostream>
#include "matrix.h"
#include "all2all.h"
#include "local.h"
#include "global.h"
#include "test.h"
#include "config.h"


using namespace std;

int main(){

    // uint mat_a[] = {
    //     100,100,100,
    //     177,142,1323,
    //     1234,34,344,
    // };

    // Matrix m(mat_a,3);
    // m.get_sdsm_info();
    // m.print();
    // FastAll2All app(&m);
    // app.to_scaled_matrix();
    // app.to_scaled_doubly_stochastic_matrix();
    // app.print();

    // Decomposition tester
    // vector<uint> test_dim = {10, 50, 100, 200};
    // DecompositionTester tester(test_dim);
    // tester.run(true);

    // uint mat_b[] = {
    //     100,100,100, 1, 2, 3, 10, 11, 12,
    //     200,200,200, 4, 5, 6, 13, 14, 15,
    //     300,300,300, 7, 8, 9, 16, 17, 18
    // };
    // LocalScheduler l(mat_b, 3, 3, 0);
    // l.load_balance();
    // l.print();

    FastAll2AllTester simulator(2, 16, 20, false, ETHER400, H100);
    // simulator.run();
    simulator.server_gpu_number_benchmark(H100, ETHER100);
    // simulator.fabric_speed_benchmark();
    
    // A100
    // simulator.topology_benchmark(&spread_out, 100, 7200);
    // // MI300
    // simulator.topology_benchmark(&intra_transfer_full_mesh, 100, 1024);
    // // MI250
    // simulator.topology_benchmark(&intra_transfer_2ring, 100, 1800);
    // // V100
    // simulator.topology_benchmark(&intra_transfer_hybrid_cude_mesh, 100, 1200);


    // simulator.skewness_benchmark();
    // simulator.transfer_size_benchmark(false, FAST, INFB, &balance_alpha_beta, "0");
    // simulator.transfer_size_benchmark(true, FAST, INFB, &balance_alpha_beta,  "1");
    // simulator.transfer_size_benchmark(true, FAST, INFB, &always_scale, "2");


    
    return 0;
}