set(TARGET llama-app)

set(LLAMA_APP_SRCS llama.cpp download.cpp)
if (LLAMA_APU_BACKEND)
    list(APPEND LLAMA_APP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../tools/apu-run/main.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../tools/apu-run/main.cpp PROPERTIES COMPILE_DEFINITIONS "LLAMA_APU_NO_MAIN")
endif()

add_executable(${TARGET} ${LLAMA_APP_SRCS})
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama)

target_link_libraries(${TARGET} PRIVATE
    llama-server-impl
    llama-cli-impl
    llama-completion-impl
    llama-bench-impl
    llama-batched-bench-impl
    llama-fit-params-impl
    llama-quantize-impl
    llama-perplexity-impl
)
if (LLAMA_APU_BACKEND)
    target_link_libraries(${TARGET} PRIVATE llama-common)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Automatically add all files from the 'licenses' directory
file(GLOB EXTRA_LICENSES "${CMAKE_SOURCE_DIR}/licenses/LICENSE-*")

foreach(FILE_PATH ${EXTRA_LICENSES})
    get_filename_component(FILE_NAME "${FILE_PATH}" NAME)
    string(REGEX REPLACE "^LICENSE-" "" NAME "${FILE_NAME}")
    license_add_file("${NAME}" "${FILE_PATH}")
endforeach()

license_generate(${TARGET})

if (LLAMA_APU_BACKEND)
    add_dependencies(${TARGET} zero_copy_model_runner_build)
    target_include_directories(${TARGET} PRIVATE ${APU_BACKEND_DIR}/include)
    target_link_libraries(${TARGET} PRIVATE
        ${APU_BACKEND_DIR}/target/release/libzero_copy_model_runner.a
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        m
    )
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE_NAME:${TARGET}>
            $<TARGET_FILE_DIR:${TARGET}>/llama-apu
        COMMENT "Creating llama-apu symlink pointing to llama"
    )
    if(LLAMA_TOOLS_INSTALL)
        install(FILES $<TARGET_FILE_DIR:${TARGET}>/llama-apu DESTINATION bin)
    endif()
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
