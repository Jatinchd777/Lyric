cmake_minimum_required(VERSION 3.20)

project(lyric LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WHISPER_DIR
    ${CMAKE_SOURCE_DIR}/whisper.cpp
)

set(WHISPER_BUILD_DIR
    ${WHISPER_DIR}/build/bin
)

add_executable(lyric
    src/main.cpp
)

target_include_directories(lyric PRIVATE
    ${WHISPER_DIR}/include
    ${WHISPER_DIR}/ggml/include
)

target_link_directories(lyric PRIVATE
    ${WHISPER_BUILD_DIR}
)

target_link_libraries(lyric PRIVATE
    whisper
)

set_target_properties(lyric PROPERTIES
    BUILD_RPATH "${WHISPER_BUILD_DIR}"
)
