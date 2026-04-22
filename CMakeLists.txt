cmake_minimum_required(VERSION 3.15)
project(BgeOnnxTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

# Find Qt5 (installed locally)
find_package(Qt5 COMPONENTS Core Network REQUIRED)

# Use official ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-1.17.1")
include_directories("${ONNXRUNTIME_DIR}/include")
link_directories("${ONNXRUNTIME_DIR}/lib")

add_executable(BgeOnnxTest
    main.cpp
)

add_executable(BgeCApiDemo
    c_api_demo.cpp
)

add_library(BgeCore SHARED
    BertTokenizer.cpp
    BgeCore.cpp
)

target_compile_definitions(BgeCore PRIVATE BGECORE_LIBRARY)

target_link_libraries(BgeCore PRIVATE
    Qt5::Core
    onnxruntime
)

target_link_libraries(BgeOnnxTest PRIVATE
    Qt5::Core
    Qt5::Network
    BgeCore
)

target_link_libraries(BgeCApiDemo PRIVATE
    BgeCore
)

# Copy ONNX DLLs to output directory
if(WIN32)
    add_custom_command(TARGET BgeCore POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:BgeCore>
    )
    add_custom_command(TARGET BgeCore POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:BgeCore>
    )

    add_custom_command(TARGET BgeCApiDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:BgeCore>
        $<TARGET_FILE_DIR:BgeCApiDemo>
    )
    add_custom_command(TARGET BgeCApiDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:BgeCApiDemo>
    )
    add_custom_command(TARGET BgeCApiDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:BgeCApiDemo>
    )

    foreach(header_file BgeCore.h BgeCoreC.h BertTokenizer.h)
        add_custom_command(TARGET BgeCore POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}"
            $<TARGET_FILE_DIR:BgeCore>
        )
        add_custom_command(TARGET BgeCApiDemo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}"
            $<TARGET_FILE_DIR:BgeCApiDemo>
        )
        add_custom_command(TARGET BgeOnnxTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}"
            $<TARGET_FILE_DIR:BgeOnnxTest>
        )
    endforeach()
endif()
