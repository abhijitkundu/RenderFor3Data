# setting a common place to put all executable files
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

set(_APPS
  visualizePLYwithViewpoint
  visualizeRandomViewpoints
  visualizeMutiObjectRandomViewpoints  
  generateMultiObjectRandomDataset
  visualizeKittiSyntheticViewpoints
  generateKittiSyntheticDataset
  renderSingleImageInfo
  visualizeSingleImageInfo
)

foreach(APP ${_APPS})
set (APP_NAME ${APP})

add_executable(${APP_NAME} ${APP}.cpp)
target_link_libraries(${APP_NAME} RenderFor3Data)
set_property(TARGET ${APP_NAME} PROPERTY FOLDER "cpp_tools")
endforeach()