CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(NOT CMAKE_BUILD_TYPE)
  # SET(CMAKE_BUILD_TYPE "DEBUG")
  SET(CMAKE_BUILD_TYPE "RELEASE")
  # SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

set(CMAKE_CXX_FLAGS "-std=gnu++0x -fPIC ${CMAKE_CXX_FLAGS} -msse4.1 -msse2 -DEIGEN_NO_DEBUG -DVIENNACL_HAVE_EIGEN -Wno-deprecated-declarations")

find_package(PythonLibs 2.7 REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

include_directories(${PYTHON_INCLUDE_DIRS})

FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost COMPONENTS python)

include_directories(ResearchML/gpucompute/libs ResearchML/gpucompute)
add_subdirectory(${CMAKE_SOURCE_DIR}/ResearchML/gpucompute ${CMAKE_CURRENT_BINARY_DIR}/gpucompute)
add_subdirectory(${CMAKE_SOURCE_DIR}/Numpy2Eigen ${CMAKE_CURRENT_BINARY_DIR}/Numpy2Eigen)

ADD_LIBRARY(mlpy SHARED mlpy.cpp)
set_target_properties(mlpy PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(mlpy Numpy2Eigen ${Boost_LIBRARIES} gpucompute)

install(TARGETS mlpy LIBRARY DESTINATION /usr/local/lib/python2.7/dist-packages/)
