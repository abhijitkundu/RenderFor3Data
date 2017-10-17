/**
 * @file visualizeRandomViewpoints.cpp
 * @brief visualizeRandomViewpoints
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
#include <fstream>
#include <iostream>

std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > load_viewpoint_distribution(
    const std::string& vp_distrib_file) {
  std::ifstream file(vp_distrib_file.c_str(), std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File from " + vp_distrib_file);
  }

  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > viewpoints;

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    Eigen::Vector4f vp;
    if (!(iss >> vp[0] >> vp[1] >> vp[2] >> vp[3])) {
      break;
    }  // error

    vp.head<3>() *= M_PI / 180.0f;

    viewpoints.push_back(vp);
  }

  file.close();
  return viewpoints;
}


namespace CuteGL {

class ViewpointBrowser : public WindowRenderViewer {
 public:

  enum Action {
    NEXT_MODEL = 100,
    PREVIOUS_MODEL,
  };

  explicit ViewpointBrowser(AbstractRenderer* renderer, const std::string& vp_file)
      : WindowRenderViewer(renderer),
        vp_index_(-1) {

    keyboard_handler_.registerKey(NEXT_MODEL, Qt::Key_Right, "Move to next frame");
    keyboard_handler_.registerKey(PREVIOUS_MODEL, Qt::Key_Left, "Move to previous frame");


    std::cout << "Loading viewpoints ..." << std::flush;
    viewpoints_ = load_viewpoint_distribution(vp_file);
    std::cout << "Loaded " << viewpoints_.size() << " viewpoints." << std::endl;
  }

 protected:
  virtual void handleKeyboardAction(int action) {
    switch (action) {
      case NEXT_MODEL:
      {
        vp_index_ = std::min(vp_index_ + 1, int(viewpoints_.size()) - 1);
        camera().extrinsics() = getExtrinsicsFromViewPoint(viewpoints_[vp_index_]);
        break;
      }
      case PREVIOUS_MODEL:
      {
        vp_index_ = std::max(vp_index_ - 1, 0);
        camera().extrinsics() = getExtrinsicsFromViewPoint(viewpoints_[vp_index_]);
        break;
      }
      default:
        WindowRenderViewer::handleKeyboardAction(action);
        break;
    }
  }

  int vp_index_;
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > viewpoints_;
};

}  // namespace CuteGL

int main(int argc, char **argv) {

  QApplication app(argc, argv);

  using namespace CuteGL;
  using namespace Eigen;

  std::string vp_file = "/home/abhijit/Workspace/RenderFor3Data/data/view_distribution/voc2012_kitti/car.txt";

  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());

  ViewpointBrowser viewer(renderer.get(), vp_file);
  viewer.setBackgroundColor(0, 0, 0);

  const int W = 960;
  const int H = 540;
  viewer.resize(W, H);

  viewer.setSceneRadius(10.0f);

  viewer.camera().extrinsics() = getExtrinsicsFromViewPoint(0.0f, 0.0f, 0.0f, 2.0f);

  viewer.showAndWaitTillExposed();

  // Set camera intrinsics
  viewer.camera().intrinsics() = getGLPerspectiveProjection(1050.0f, 1050.0f, 0.0f, 480.0f, 270.0f, 960, 540, 0.1f, 100.0f);


  renderer->modelDrawers().addItem(Affine3f::Identity(), loadMeshFromPLY(CUTEGL_ASSETS_FOLDER "/car.ply"));
  renderer->modelDrawers().poses().front().setIdentity();

  app.exec();
}

