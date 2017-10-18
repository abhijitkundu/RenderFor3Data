CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)
PROJECT(RenderFor3Data)

########################### Misc. Configs ##############################
# Add custom cmake files folder
SET (PROJECT_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)
SET (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_CMAKE_DIR}")

# Add custom Compile Settings and flags
INCLUDE(CompileSettings)

##################### Look for required libraries ######################
MESSAGE(STATUS "===============================================================")
MESSAGE(STATUS "============= Look for required libraries =====================")

FIND_PACKAGE(OpenCV REQUIRED)
IF (${OpenCV_VERSION_MAJOR} EQUAL 2)
  SET(REQUIRED_OpenCV_LIBS opencv_core opencv_highgui)
ELSE()
  SET(REQUIRED_OpenCV_LIBS opencv_core opencv_highgui opencv_imgcodecs)
ENDIF()

# Find HDF5 dependencies
FIND_PACKAGE(HDF5 1.8.12
  REQUIRED
  COMPONENTS
    CXX
  )

# Add Boost dependencies
FIND_PACKAGE(Boost
  REQUIRED
  COMPONENTS
    filesystem
    system
    program_options
    serialization
    timer
  )

# Find QT Dependencies
FIND_PACKAGE(Qt5Widgets REQUIRED)

# Find CuteGL Dependencies
FIND_PACKAGE(CuteGL REQUIRED)

# Find Eigen dependencies
FIND_PACKAGE(Eigen REQUIRED)

# Find JSON Dependencies
find_package(nlohmann_json REQUIRED)

message(STATUS "=============================================================")
message(STATUS "============ Generating RenderFor3DataConfig.h ==============")

set(RENDERFOR3DATA_ROOT_DIR \"${CMAKE_SOURCE_DIR}\")
configure_file( ${CMAKE_SOURCE_DIR}/cmake/Config.h.cmake ${CMAKE_SOURCE_DIR}/RenderFor3Data/Config.h )

MESSAGE(STATUS "===============================================================")
MESSAGE(STATUS "==============  Adding RenderFor3Data Library  ================")
ADD_SUBDIRECTORY(RenderFor3Data)

MESSAGE(STATUS "===============================================================")
MESSAGE(STATUS "=================   Adding CPP Tools   ========================")
ADD_SUBDIRECTORY(cpp_tools)
  
########################## Display Summary #############################

MESSAGE(STATUS "===============================================================")
MESSAGE(STATUS "================  Configuration Summary  ======================")
MESSAGE(STATUS "Project Name:       ${PROJECT_NAME}")
MESSAGE(STATUS "C++ Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
MESSAGE(STATUS "C++ Standard:       C++ ${CMAKE_CXX_STANDARD}")
MESSAGE(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Build type Flags:   ${CMAKE_BUILD_TYPE_FLAGS}")
MESSAGE(STATUS "C++ compile flags:  ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "Install Path:       ${CMAKE_INSTALL_PREFIX}")
MESSAGE(STATUS "Eigen Version:      ${EIGEN_VERSION}")
MESSAGE(STATUS "HDF5 Version:       ${HDF5_VERSION}")
MESSAGE(STATUS "Boost Version:      ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
MESSAGE(STATUS "OpenCV Version:     ${OpenCV_VERSION}")
message(STATUS "Json:               ${nlohmann_json_VERSION}")
MESSAGE(STATUS "===============================================================")
