cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Correction")

# 1. OpenCV setup
set(OpenCV_DIR "C:/c++ libarys/opencv/build")
find_package(OpenCV REQUIRED)

# 2. OpenVINO setup
set(OPENVINO_ROOT "C:/c++ libarys/openvino/runtime")
include_directories("${OPENVINO_ROOT}/include")
link_directories("${OPENVINO_ROOT}/lib/intel64/Release")

# 3. Add source to this project's executable
add_executable(Correction "Correction.cpp" "Correction.h")

# 4. Link OpenCV libraries
target_include_directories(Correction PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Correction PRIVATE ${OpenCV_LIBS})

# 5. Link OpenVINO libraries
# Adjust plugin libraries if needed (CPU, GPU, MYRIAD, etc.)
target_link_libraries(Correction PRIVATE openvino)

# 6. Set C++ standard
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Correction PROPERTY CXX_STANDARD 20)
endif()