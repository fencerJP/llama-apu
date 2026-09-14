# SPDX-License-Identifier: Apache-2.0
# apu-run: Native AMD Ryzen AI APU Zero-Copy runner for llama.cpp

set(TARGET apu-run)

if (NOT DEFINED APU_BACKEND_DIR)
    set(APU_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../zero-copy_model_runner")
endif()

add_executable(${TARGET} main.cpp)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${APU_BACKEND_DIR}/include
)

target_link_libraries(${TARGET} PRIVATE
    llama-common
    llama
    ${APU_BACKEND_DIR}/target/release/libzero_copy_model_runner.a
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    m
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
