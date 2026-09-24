#include "ggml-apu-xclbin.h"
#include <cstdio>

int main(int argc, char ** argv) {
    bool verbose = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-q" || std::string(argv[i]) == "--quiet") {
            verbose = false;
        }
    }

    printf("Running APU Phase 7.1 XCLBIN Synthesis Unit Tests...\n");
    bool ok = test_apu_xclbin_synth(verbose);
    if (!ok) {
        fprintf(stderr, "test-apu-xclbin: FAILED\n");
        return 1;
    }

    printf("test-apu-xclbin: ALL TESTS PASSED\n");
    return 0;
}
