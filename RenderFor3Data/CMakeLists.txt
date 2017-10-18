file(GLOB _HDRS *.h *.hpp)
file(GLOB _SRCS *.cpp)

########################### Add Target ################################

# # Instruct CMake to run moc automatically when needed.
# set(CMAKE_AUTOMOC ON)

add_library(RenderFor3Data ${_SRCS} ${_HDRS})
target_link_libraries(RenderFor3Data
  CuteGL::Surface
  nlohmann_json
  ${Boost_LIBRARIES}
  ${HDF5_LIBRARIES}  
  ${OpenCV_LIBS}
  )

target_include_directories(RenderFor3Data
  PUBLIC
    ${EIGEN_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
 )

target_include_directories(RenderFor3Data
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_SOURCE_DIR}
 )