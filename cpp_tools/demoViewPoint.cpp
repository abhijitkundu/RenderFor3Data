/**
 * @file demoViewPoint.cpp
 * @brief demoViewPoint
 *
 * @author Abhijit Kundu
 */

#include "CuteGL/Renderer/MultiObjectRenderer.h"
#include "CuteGL/Surface/WindowRenderViewer.h"
#include "CuteGL/Core/Config.h"
#include "CuteGL/IO/ImportPLY.h"
#include "CuteGL/Core/PoseUtils.h"

#include <Eigen/EulerAngles>

#include <QApplication>
#include <iostream>

int main(int argc, char **argv) {

  QApplication app(argc, argv);

  using namespace CuteGL;
  using namespace Eigen;

  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());

  WindowRenderViewer viewer(renderer.get());
  viewer.setBackgroundColor(0, 0, 0);

  const int W = 1024;
  const int H = 768;
  viewer.resize(W, H);

  viewer.setSceneRadius(10.0f);

  viewer.camera().extrinsics() = getExtrinsicsFromViewPoint(0.0f, 0.0f, 0.0f, 2.0f);

  viewer.showAndWaitTillExposed();

  renderer->modelDrawers().addItem(Affine3f::Identity(), loadMeshFromPLY(CUTEGL_ASSETS_FOLDER "/car.ply"));
  
  renderer->modelDrawers().poses().front().setIdentity();
  Vector3f viewpoint = Vector3f(20.0f, 10.0f, 0.0f) * M_PI / 180.0f;
  viewer.camera().extrinsics() = getExtrinsicsFromViewPoint(viewpoint.x(), viewpoint.y(), viewpoint.z(), 2.0f);

  app.exec();
}


