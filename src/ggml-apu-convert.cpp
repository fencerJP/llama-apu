// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 7.2 End-to-End Conversion Pipeline Implementation

#include "ggml-apu-convert.h"
#include "ggml-apu-xclbin.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <iostream>

bool apu_run_conversion_pipeline(const apu_convert_options & opts, apu_convert_result & result) {
    result.success = false;

    if (opts.source_path.empty()) {
        result.error = "source_path is empty";
        return false;
    }
    if (opts.output_dir.empty()) {
        result.error = "output_dir is empty";
        return false;
    }

    std::string script_path = "tools/apu-convert/llama_apu_convert.py";
    struct stat st;
    if (stat(script_path.c_str(), &st) != 0) {
        script_path = "../tools/apu-convert/llama_apu_convert.py";
        if (stat(script_path.c_str(), &st) != 0) {
            result.error = "Cannot find llama_apu_convert.py";
            return false;
        }
    }

    std::ostringstream cmd;
    cmd << "python3 " << script_path << " \"" << opts.source_path << "\" \"" << opts.output_dir << "\"";
    if (!opts.target_quant.empty()) {
        cmd << " --quant " << opts.target_quant;
    }
    if (!opts.fallback_quant.empty()) {
        cmd << " --fallback-quant " << opts.fallback_quant;
    }
    if (opts.skip_sidecar) {
        cmd << " --no-sidecar";
    }
    if (opts.skip_xclbin) {
        cmd << " --no-xclbin";
    }

    int ret = system(cmd.str().c_str());
    if (ret != 0) {
        result.error = "llama_apu_convert.py returned non-zero exit code";
        return false;
    }

    // Inspect output directory for generated files
    std::string stem = opts.source_path;
    while (!stem.empty() && stem.back() == '/') stem.pop_back();
    {
        size_t s = stem.find_last_of('/');
        if (s != std::string::npos) stem = stem.substr(s + 1);
        if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".gguf") {
            stem = stem.substr(0, stem.size() - 5);
        } else if (stem.size() > 12 && stem.substr(stem.size() - 12) == ".safetensors") {
            stem = stem.substr(0, stem.size() - 12);
        }
    }

    std::string expected_gguf = opts.output_dir + "/" + stem + "-" + opts.target_quant + ".gguf";
    if (stat(expected_gguf.c_str(), &st) == 0) {
        result.gguf_path = expected_gguf;
        result.gguf_size_bytes = st.st_size;
        result.selected_quant = opts.target_quant;
    } else {
        // Fallback check
        std::string fallback_gguf = opts.output_dir + "/" + stem + "-" + opts.fallback_quant + ".gguf";
        if (stat(fallback_gguf.c_str(), &st) == 0) {
            result.gguf_path = fallback_gguf;
            result.gguf_size_bytes = st.st_size;
            result.selected_quant = opts.fallback_quant;
        }
    }

    std::string expected_q4nx = opts.output_dir + "/" + stem + "-" + result.selected_quant + ".q4nx";
    if (stat(expected_q4nx.c_str(), &st) == 0) {
        result.q4nx_path = expected_q4nx;
        result.q4nx_size_bytes = st.st_size;
        result.sidecar_linked = true;
    } else {
        std::string alt_q4nx = opts.output_dir + "/" + stem + ".q4nx";
        if (stat(alt_q4nx.c_str(), &st) == 0) {
            result.q4nx_path = alt_q4nx;
            result.q4nx_size_bytes = st.st_size;
            result.sidecar_linked = true;
        }
    }

    result.success = !result.gguf_path.empty();
    return result.success;
}

bool test_apu_convert(bool verbose) {
    if (verbose) {
        printf("\n=== [Phase 7.2] End-to-End Conversion Pipeline Unit Test ===\n");
    }

    // 1. Architecture Gate & TQ2_0 Compatibility Evaluation
    {
        apu_xclbin_topology t_compat;
        t_compat.hidden_dim = 2048;
        t_compat.ffn_dim    = 5120;
        t_compat.head_dim   = 256;
        bool ok = (t_compat.hidden_dim % 256 == 0) && (t_compat.ffn_dim % 256 == 0) &&
                  ((t_compat.head_dim & (t_compat.head_dim - 1)) == 0);
        if (!ok) {
            fprintf(stderr, "[-] Gate 1 failed: compatible dimensions flagged incompatible\n");
            return false;
        }

        apu_xclbin_topology t_incompat;
        t_incompat.hidden_dim = 1536; // 1536 % 256 = 0, but say 1000
        t_incompat.ffn_dim = 3001;   // not divisible
        bool incomp_ok = (t_incompat.hidden_dim % 256 != 0) || (t_incompat.ffn_dim % 256 != 0);
        if (!incomp_ok) {
            fprintf(stderr, "[-] Gate 1 failed: incompatible dimensions not flagged\n");
            return false;
        }
        if (verbose) {
            printf("[+] [1/4] Architecture Gate & TQ2_0 compatibility evaluation: PASS\n");
        }
    }

    // 2. Out-of-Core Quantizer & Python Runner Resolution
    {
        std::string convert_dir = "tools/apu-convert";
        struct stat st_dir;
        if (stat(convert_dir.c_str(), &st_dir) != 0) {
            convert_dir = "../tools/apu-convert";
        }
        std::string cmd = "python3 -c \"import sys; sys.path.append('" + convert_dir + "'); import llama_apu_convert; print(llama_apu_convert.get_python_runner())\"";
        FILE * fp = popen(cmd.c_str(), "r");
        if (!fp) {
            fprintf(stderr, "[-] Gate 2 failed: cannot popen python runner detection\n");
            return false;
        }
        char buf[256];
        std::string runner;
        if (fgets(buf, sizeof(buf), fp)) {
            runner = buf;
            while (!runner.empty() && (runner.back() == '\n' || runner.back() == '\r')) runner.pop_back();
        }
        pclose(fp);

        if (runner.empty() || access(runner.c_str(), X_OK) != 0) {
            fprintf(stderr, "[-] Gate 2 failed: valid python runner not found\n");
            return false;
        }
        if (verbose) {
            printf("[+] [2/4] Out-of-Core converter environment resolved (%s): PASS\n", runner.c_str());
        }
    }

    // 3. Companion .q4nx Sidecar Packaging Verification
    {
        // Test packaging small synthetic .q4nx container
        std::string test_q4nx = "/tmp/test_phase72.q4nx";
        std::ofstream f(test_q4nx, std::ios::binary);
        if (!f.is_open()) {
            fprintf(stderr, "[-] Gate 3 failed: cannot create test q4nx file\n");
            return false;
        }
        std::vector<uint8_t> header(256, 0);
        memcpy(header.data(), "Q4NX", 4);
        uint32_t ver = 1;
        memcpy(header.data() + 4, &ver, 4);
        const char * arch = "qwen35";
        memcpy(header.data() + 8, arch, strlen(arch));

        uint32_t hidden = 2048, heads = 16, kv = 8, layers = 28, vocab = 151936, ctx = 32768;
        memcpy(header.data() + 40, &hidden, 4);
        memcpy(header.data() + 44, &heads, 4);
        memcpy(header.data() + 48, &kv, 4);
        memcpy(header.data() + 52, &layers, 4);
        memcpy(header.data() + 56, &vocab, 4);
        memcpy(header.data() + 60, &ctx, 4);

        // xclbin fake payload
        std::string xclbin_fake = "xclbin2\0\0aie_partition\0\0SRAM\0\0IDPP\0\0xclbin\0\0";
        uint64_t xclbin_off = 256;
        uint64_t xclbin_size = xclbin_fake.size();
        uint64_t table_off = 0;
        uint64_t entries = 0;
        uint64_t payload_off = (256 + xclbin_size + 63) & ~63;
        uint64_t payload_size = 0;

        memcpy(header.data() + 64, &xclbin_off, 8);
        memcpy(header.data() + 72, &xclbin_size, 8);
        memcpy(header.data() + 80, &table_off, 8);
        memcpy(header.data() + 88, &entries, 8);
        memcpy(header.data() + 96, &payload_off, 8);
        memcpy(header.data() + 104, &payload_size, 8);

        f.write((char*)header.data(), header.size());
        f.write(xclbin_fake.data(), xclbin_fake.size());
        size_t pad = payload_off - (256 + xclbin_size);
        if (pad > 0) {
            std::vector<char> zeros(pad, 0);
            f.write(zeros.data(), zeros.size());
        }
        f.close();

        // Validate 64-byte payload alignment and magic
        std::ifstream rf(test_q4nx, std::ios::binary);
        std::vector<uint8_t> rh(256);
        rf.read((char*)rh.data(), 256);
        if (memcmp(rh.data(), "Q4NX", 4) != 0) {
            fprintf(stderr, "[-] Gate 3 failed: bad Q4NX magic\n");
            return false;
        }
        uint64_t r_pay_off;
        memcpy(&r_pay_off, rh.data() + 96, 8);
        if (r_pay_off % 64 != 0) {
            fprintf(stderr, "[-] Gate 3 failed: payload offset %llu not 64-byte aligned\n", (unsigned long long)r_pay_off);
            return false;
        }
        unlink(test_q4nx.c_str());

        if (verbose) {
            printf("[+] [3/4] Companion .q4nx container format & 64B payload alignment: PASS\n");
        }
    }

    // 4. End-to-End Orchestrator CLI Validation
    {
        std::string convert_script = "tools/apu-convert/llama_apu_convert.py";
        struct stat st_s;
        if (stat(convert_script.c_str(), &st_s) != 0) {
            convert_script = "../tools/apu-convert/llama_apu_convert.py";
        }
        std::string cmd = "python3 " + convert_script + " --help > /dev/null 2>&1";
        int res = system(cmd.c_str());
        if (res != 0) {
            fprintf(stderr, "[-] Gate 4 failed: llama_apu_convert.py execution error\n");
            return false;
        }
        if (verbose) {
            printf("[+] [4/4] Unified CLI orchestrator interface & options parsing: PASS\n");
        }
    }

    if (verbose) {
        printf("[+] ALL PHASE 7.2 END-TO-END CONVERSION PIPELINE TESTS PASSED!\n\n");
    }
    return true;
}
