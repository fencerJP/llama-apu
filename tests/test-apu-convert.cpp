// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 7.2 End-to-End Conversion Pipeline Unit Test Driver

#include "ggml-apu-convert.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    bool verbose = true;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-q") || !strcmp(argv[i], "--quiet")) {
            verbose = false;
        }
    }

    printf("Running APU Phase 7.2 End-to-End Conversion Pipeline Unit Tests...\n");
    if (!test_apu_convert(verbose)) {
        fprintf(stderr, "test-apu-convert: FAILED\n");
        return 1;
    }
    printf("test-apu-convert: ALL TESTS PASSED\n");
    return 0;
}
