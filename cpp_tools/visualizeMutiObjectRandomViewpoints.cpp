/**
 * @file visualizeMutiObjectRandomViewpoints.cpp
 * @brief visualizeMutiObjectRandomViewpoints
 *
 * @author Abhijit Kundu
 */

#include "RenderFor3DataConfig.h"
#include <CuteGL/Renderer/MultiObjectRenderer.h>
#include <CuteGL/Surface/WindowRenderViewer.h>
#include <CuteGL/Core/Config.h>
#include <CuteGL/IO/ImportPLY.h>
#include <CuteGL/Core/PoseUtils.h>
#include <CuteGL/Core/MeshUtils.h>
#include <CuteGL/Core/ColorUtils.h>
#include <CuteGL/Geometry/OrientedBoxHelper.h>

#include <Eigen/EulerAngles>
#include <boost/filesystem.hpp>
#include <QApplication>
#include <random>
#include <fstream>
#include <iostream>

namespace CuteGL {

class ViewpointBrowser : public WindowRenderViewer {
 public:

  enum Action {
    NEXT_IMAGE = 100,
  };

  explicit ViewpointBrowser(MultiObjectRenderer* renderer, const std::size_t num_of_objects_per_image)
      : WindowRenderViewer(renderer),
        renderer_(renderer),
        num_of_objects_per_image_(num_of_objects_per_image),
        vp_index_(0),
        model_index_(0),
        rnd_eng_ { std::random_device { }() },
        K_(Eigen::Matrix3f::Identity()){

    keyboard_handler_.registerKey(NEXT_IMAGE, Qt::Key_Right, "Next image");
    keyboard_handler_.registerKey(NEXT_IMAGE, Qt::Key_Left, "Next image");
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


  std::size_t readViewpoints(const std::string& vp_distrib_filepath) {
    std::ifstream file(vp_distrib_filepath.c_str(), std::ios::in);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open File from " + vp_distrib_filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      Eigen::Vector3f vp;
      if (!(iss >> vp[0] >> vp[1] >> vp[2])) {
        break;
      }  // error

      vp *= M_PI / 180.0f;

      viewpoints_.push_back(vp);
    }

    file.close();

    vp_indices_.resize(viewpoints_.size());
    std::iota(vp_indices_.begin(), vp_indices_.end(), 0);
    std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);

    return vp_indices_.size();
  }

  std::size_t readModelFilesList(const std::string& models_list_filepath, const std::string& prefix, const std::string& suffix) {
    std::ifstream file(models_list_filepath.c_str(), std::ios::in);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open File from " + models_list_filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
      model_filepaths_.push_back(prefix + line + suffix);
    }

    file.close();

    model_indices_.resize(model_filepaths_.size());
    std::iota(model_indices_.begin(), model_indices_.end(), 0);
    std::shuffle(model_indices_.begin(), model_indices_.end(), rnd_eng_);

    return model_indices_.size();
  }

 protected:

  void loadModels() {
    assert(model_filepaths_.size() == model_indices_.size());
    assert(num_of_objects_per_image_ <= model_indices_.size());
    renderer_->modelDrawers().clear();

    std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> sat_dist(0.95f, 1.0f);
    std::uniform_real_distribution<float> val_dist(0.95f, 1.0f);


    for (std::size_t i = 0; i < num_of_objects_per_image_; ++i) {
      ++model_index_;
      if (model_index_ >= model_indices_.size()) {
        model_index_ = 0;
        std::shuffle(model_indices_.begin(), model_indices_.end(), rnd_eng_);
      }

      auto mesh = loadMeshFromPLY(model_filepaths_.at(model_indices_.at(model_index_)));
      const float golden_ratio_conjugate = 0.618033988749895f;
      const float hue = 360.0f * std::fmod(hue_dist(rnd_eng_) + golden_ratio_conjugate, 1.0f);
      const MeshData::ColorType color = CuteGL::makeRGBAfromHSV(hue, sat_dist(rnd_eng_), val_dist(rnd_eng_));
      CuteGL::colorizeMesh(mesh, color);
      renderer_->modelDrawers().addItem(Eigen::Affine3f::Identity(), mesh);
    }

    assert(model_filepaths_.size() == model_indices_.size());
  }

  void setModelPoses() {
    assert(viewpoints_.size() == vp_indices_.size());
    std::uniform_real_distribution<float> x_dis(0.0f, width());
    std::uniform_real_distribution<float> y_dis(0.0f, height());
    std::uniform_real_distribution<float> z_dis(1.0f, 30.0f);

    MultiObjectRenderer::ModelDrawers::Poses& poses = renderer_->modelDrawers().poses();
    const Eigen::Matrix3f Kinv = K_.inverse();
    const std::size_t number_of_objects = renderer_->modelDrawers().poses().size();

    for (size_t i = 0; i < number_of_objects; ++i) {
      ++vp_index_;
      if (vp_index_ >= vp_indices_.size()) {
        vp_index_ = 0;
        std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);
      }
      const Eigen::Vector3f& vp = viewpoints_[vp_indices_[vp_index_]];

      // Loop and keep sampling new center_proj until we find a collision free pose
      while (true) {
        Eigen::Isometry3f vp_pose = getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), z_dis(rnd_eng_));
        Eigen::Vector3f center_proj_ray = Kinv * Eigen::Vector3f(x_dis(rnd_eng_), y_dis(rnd_eng_), 1.0f);
        Eigen::Isometry3f pose = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), center_proj_ray) * vp_pose;

        bool collision_free = true;

        // check for all previous poses for collision
        for (size_t j = 0; j < i; ++j) {
          if (checkOrientedBoxCollision(pose, Eigen::Vector3f::Ones(), poses[j], Eigen::Vector3f::Ones())) {
            collision_free = false;
            break;
          }
        }

        if (collision_free) {
          poses[i] = pose;
          break;
        }
      }
    }
  }


  virtual void handleKeyboardAction(int action) {
    switch (action) {
      case NEXT_IMAGE: {
        loadModels();
        setModelPoses();
        break;
      }
      default:
        WindowRenderViewer::handleKeyboardAction(action);
        break;
    }
  }

  MultiObjectRenderer* renderer_;

  std::size_t num_of_objects_per_image_;

  std::vector<Eigen::Vector3f> viewpoints_;
  std::vector<std::size_t> vp_indices_;
  std::size_t vp_index_;

  std::vector<std::string> model_filepaths_;
  std::vector<std::size_t> model_indices_;
  std::size_t model_index_;


  std::mt19937 rnd_eng_;
  Eigen::Matrix3f K_;
};

}  // namespace CuteGL

int main(int argc, char **argv) {

  QApplication app(argc, argv);

  using namespace CuteGL;
  using namespace Eigen;

  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());
  renderer->setDisplayAxis(false);

  ViewpointBrowser viewer(renderer.get(), 32);
  viewer.setBackgroundColor(0, 0, 0);

  //  const int W = 960;
  //  const int H = 540;
  //  const float focal_length = 1050.0f;

  const int W = 1600;
  const int H = 800;
  const float focal_length = 1750.0f;

  viewer.resize(W, H);
  viewer.setSceneRadius(50.0f);

  viewer.camera().extrinsics() = Isometry3f::Identity();
  viewer.showAndWaitTillExposed();

  renderer->phongShader().program.bind();
  renderer->phongShader().setLightPosition(0.0f, -50.0f, 10.0f);
  renderer->phongShader().program.release();

  // Set camera intrinsics
  viewer.setCameraIntrinsics(focal_length, focal_length, 0.0f, W / 2.0f, H / 2.0f, W, H, 0.1f, 100.0f);

  std::cout << "Reading viewpoints ..." << std::flush;
  std::size_t num_of_vps = viewer.readViewpoints(RENDERFOR3DATA_ROOT_DIR "/data/view_distribution/voc2012_kitti/car.txt");
  std::cout << "We now have " << num_of_vps << " viewpoints." << std::endl;

  std::cout << "Reading model filelist ..." << std::flush;
  std::size_t num_of_models = viewer.readModelFilesList(RENDERFOR3DATA_ROOT_DIR "/data/ShapeNetCore_v1_clean_cars.txt",
                                                 RENDERFOR3DATA_ROOT_DIR "/data/ShapeNetCore_v1_PLY/Cars/", ".ply");
  std::cout << "We now have " << num_of_models << " models." << std::endl;

  app.exec();
}

