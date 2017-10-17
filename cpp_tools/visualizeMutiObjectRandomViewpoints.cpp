/**
 * @file visualizeMutiObjectRandomViewpoints.cpp
 * @brief visualizeMutiObjectRandomViewpoints
 *
 * @author Abhijit Kundu
 */

#include <CuteGL/Renderer/MultiObjectRenderer.h>
#include <CuteGL/Surface/WindowRenderViewer.h>
#include <CuteGL/Core/Config.h>
#include <CuteGL/IO/ImportPLY.h>
#include <CuteGL/Core/PoseUtils.h>
#include <CuteGL/Core/MeshUtils.h>
#include <CuteGL/Core/ColorUtils.h>

#include <Eigen/EulerAngles>

#include <QApplication>
#include <random>
#include <fstream>
#include <iostream>

std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > load_viewpoint_distribution(
    const std::string& vp_distrib_file) {
  std::ifstream file(vp_distrib_file.c_str(), std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File from " + vp_distrib_file);
  }

  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > viewpoints;

  std::mt19937 gen(std::random_device { }());  //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dis(1.0f, 30.0f);

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    Eigen::Vector4f vp;
    if (!(iss >> vp[0] >> vp[1] >> vp[2] >> vp[3])) {
      break;
    }  // error

    vp[3] = dis(gen);

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
    NEXT_SAMPLE = 100,
  };

  explicit ViewpointBrowser(MultiObjectRenderer* renderer, const std::string& vp_file)
      : WindowRenderViewer(renderer),
        renderer_(renderer),
        vp_index_(0),
        rnd_eng_ { std::random_device { }() },
        K_(Eigen::Matrix3f::Identity()){

    keyboard_handler_.registerKey(NEXT_SAMPLE, Qt::Key_Right, "Next sample");
    keyboard_handler_.registerKey(NEXT_SAMPLE, Qt::Key_Left, "Next sample");

    std::cout << "Loading viewpoints ..." << std::flush;
    viewpoints_ = load_viewpoint_distribution(vp_file);
    std::cout << "Loaded " << viewpoints_.size() << " viewpoints." << std::endl;

    vp_indices_.resize(viewpoints_.size());
    std::iota(vp_indices_.begin(), vp_indices_.end(), 0);
    std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);
  }

  void setCameraIntrinsics(float fx, float fy,
                           float skew,
                           float cx, float cy,
                           int img_width, int img_height,
                           float near_z, float far_z) {
    K_ << fx, skew, cx,
           0,   fy, cy,
           0,    0,  1;
    std::cout << "K=\n" << K_ << "\n";

    resize(img_width, img_height);
    camera().intrinsics() = CuteGL::getGLPerspectiveProjection(fx, fy, 0.0f, cx, cy, img_width, img_height, near_z, far_z);
  }

 protected:
  virtual void handleKeyboardAction(int action) {
    switch (action) {
      case NEXT_SAMPLE: {

        std::uniform_real_distribution<float> x_dis(0.0f, width());
        std::uniform_real_distribution<float> y_dis(0.0f, height());

        const Eigen::Matrix3f Kinv = K_.inverse();

        for (auto& pose : renderer_->modelDrawers().poses()) {
          ++vp_index_;
          if (vp_index_ >= vp_indices_.size()) {
            vp_index_ = 0;
            std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);
          }
          Eigen::Isometry3f vp_pose = getExtrinsicsFromViewPoint(viewpoints_[vp_indices_[vp_index_]]);

          Eigen::Vector3f center_proj_ray = Kinv * Eigen::Vector3f(x_dis(rnd_eng_), y_dis(rnd_eng_), 1.0f);

          pose = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), center_proj_ray) * vp_pose;
        }
        break;
      }
      default:
        WindowRenderViewer::handleKeyboardAction(action);
        break;
    }
  }
  MultiObjectRenderer* renderer_;
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > viewpoints_;
  std::vector<std::size_t> vp_indices_;
  std::size_t vp_index_;
  std::mt19937 rnd_eng_;
  Eigen::Matrix3f K_;
};

}  // namespace CuteGL

int main(int argc, char **argv) {

  QApplication app(argc, argv);

  using namespace CuteGL;
  using namespace Eigen;

  std::string vp_file = "/home/abhijit/Workspace/RenderFor3Data/data/view_distribution/voc2012_kitti/car.txt";
  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());
  renderer->setDisplayAxis(false);

  ViewpointBrowser viewer(renderer.get(), vp_file);
  viewer.setBackgroundColor(0, 0, 0);

  //  const int W = 960;
  //  const int H = 540;
  //  const float focal_length = 1050.0f;

  const int W = 1600;
  const int H = 800;
  const float focal_length = 1750.0f;

  viewer.resize(W, H);
  viewer.setSceneRadius(3.0f);

  viewer.camera().extrinsics() = Isometry3f::Identity();
  viewer.showAndWaitTillExposed();

  renderer->phongShader().program.bind();
  renderer->phongShader().setLightPosition(0.0f, -50.0f, 10.0f);
  renderer->phongShader().program.release();

  // Set camera intrinsics
  viewer.setCameraIntrinsics(focal_length, focal_length, 0.0f, W / 2.0f, H / 2.0f, W, H, 0.1f, 100.0f);

  {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> sat_dist(0.95f, 1.0f);
    std::uniform_real_distribution<float> val_dist(0.95f, 1.0f);

    auto mesh = loadMeshFromPLY(CUTEGL_ASSETS_FOLDER "/car.ply");

    const int num_of_meshes = 32;
    for (int i = 0; i < num_of_meshes; ++i) {
      const float golden_ratio_conjugate = 0.618033988749895f;
      const float hue = 360.0f * std::fmod(hue_dist(gen) + golden_ratio_conjugate, 1.0f);
      const MeshData::ColorType color = CuteGL::makeRGBAfromHSV(hue, sat_dist(gen), val_dist(gen));
      CuteGL::colorizeMesh(mesh, color);
      renderer->modelDrawers().addItem(Affine3f::Identity(), mesh);
    }

  }

  app.exec();
}

