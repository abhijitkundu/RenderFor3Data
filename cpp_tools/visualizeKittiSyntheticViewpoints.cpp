/**
 * @file visualizeKittiSyntheticViewpoints.cpp
 * @brief visualizeKittiSyntheticViewpoints
 *
 * @author Abhijit Kundu
 */

#include "RenderFor3Data/Config.h"
#include "RenderFor3Data/ImageDataset.h"
#include <CuteGL/Renderer/MultiObjectRenderer.h>
#include <CuteGL/Surface/WindowRenderViewer.h>
#include <CuteGL/Core/Config.h>
#include <CuteGL/IO/ImportPLY.h>
#include <CuteGL/Core/PoseUtils.h>
#include <CuteGL/Core/MeshUtils.h>
#include <CuteGL/Core/ColorUtils.h>
#include <CuteGL/Geometry/ComputeAlignedBox.h>
#include <CuteGL/Geometry/OrientedBoxHelper.h>

#include <Eigen/EulerAngles>
#include <boost/filesystem.hpp>
#include <QApplication>
#include <random>
#include <fstream>
#include <iostream>

Eigen::Isometry3d pose_from_obj_info(const ImageObjectInfo& obj_info, const Eigen::Matrix3d& K_inv) {
  Eigen::Vector3d vp = obj_info.viewpoint.value();
  Eigen::Isometry3d vp_pose = CuteGL::getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), obj_info.center_dist.value());
  Eigen::Vector3d center_proj_ray = K_inv * obj_info.center_proj.value().homogeneous();
  Eigen::Isometry3d pose = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), center_proj_ray) * vp_pose;
  return pose;
}

namespace CuteGL {

class ViewpointBrowser : public WindowRenderViewer {
 public:

  enum Action {
    NEXT_IMAGE = 100,
    PREV_IMAGE = 101,
  };

  explicit ViewpointBrowser(MultiObjectRenderer* renderer, const std::size_t num_of_objects_per_image, const std::string& kitti_gt_json_file)
      : WindowRenderViewer(renderer),
        renderer_(renderer),
        num_of_objects_per_image_(num_of_objects_per_image),
        img_index_(0),
        model_index_(0),
        rnd_eng_ {42} {

    keyboard_handler_.registerKey(NEXT_IMAGE, Qt::Key_Right, "Next image");
    keyboard_handler_.registerKey(PREV_IMAGE, Qt::Key_Left, "Prev image");

    image_dataset_ = loadImageDatasetFromJson(kitti_gt_json_file);
    for (const auto& image_info : image_dataset_.image_infos) {
      for (const auto& obj_info : image_info.object_infos.value()) {
        all_object_infos_.push_back(obj_info);
      }
    }
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

  void update() {
    sampleCurrentImageObjectInfos();
    visualizeCurrentImageObjectInfos();
  }

 protected:

  void setCameraIntrinsics(const Eigen::Matrix3d& K, const Eigen::Vector2i& image_size) {
    resize(image_size[0], image_size[1]);
    camera().intrinsics() = getGLPerspectiveProjection(K.cast<float>(), image_size[0], image_size[1], 0.1f, 100.0f);
  }

  void sampleCurrentImageObjectInfos() {
    cb_obj_infos_.clear();

    const ImageInfo& img_info = image_dataset_.image_infos.at(img_index_);

    const Eigen::Matrix3d K = img_info.image_intrinsic.value();
    const Eigen::Matrix3d K_inv = K.inverse();

    setCameraIntrinsics(K, img_info.image_size.value());

    using Poses = std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>;
    Poses cb_poses;
    cb_poses.reserve(num_of_objects_per_image_);


    const ImageInfo::ImageObjectInfos& current_image_obj_infos = img_info.object_infos.value();
    // Populate as much as possible from current image
    for (size_t i = 0; i < std::min(num_of_objects_per_image_, current_image_obj_infos.size()); ++i) {
      cb_obj_infos_.push_back(current_image_obj_infos[i]);
      cb_poses.push_back(pose_from_obj_info(current_image_obj_infos[i], K_inv));
    }

    // Fill the rest by random sampling and jittering
    std::uniform_int_distribution<size_t> obj_dis(0, all_object_infos_.size() - 1);
    std::uniform_real_distribution<double> azimuth_delta_dis(-10.0, 10.0);
    std::uniform_real_distribution<double> elevation_delta_dis(-1.0, 1.0);
    std::uniform_real_distribution<double> tilt_delta_dis(-1.0, 1.0);


    for (size_t i = cb_obj_infos_.size(); i < num_of_objects_per_image_; ++i) {
      while (true) {
        ImageObjectInfo random_obj_info = all_object_infos_.at(obj_dis(rnd_eng_));
        Eigen::Vector3d& vp = random_obj_info.viewpoint.value();
        vp += Eigen::Vector3d(azimuth_delta_dis(rnd_eng_), elevation_delta_dis(rnd_eng_), tilt_delta_dis(rnd_eng_)) * M_PI / 180.0;
        //TODO re wrap to pi

        Eigen::Isometry3d pose = pose_from_obj_info(random_obj_info, K_inv);

        bool collision_free = true;
        // check for all previous poses for collision
        for (size_t j = 0; j < i; ++j) {
          ImageObjectInfo& obj_info_j = cb_obj_infos_.at(j);
          if (CuteGL::checkOrientedBoxCollision(pose, random_obj_info.dimension.value(), cb_poses.at(j), obj_info_j.dimension.value())) {
            collision_free = false;
            break;
          }
        }

        if (collision_free) {
          cb_obj_infos_.push_back(random_obj_info);
          cb_poses.push_back(pose);
          break;
        }
      }

    }
  }


  void visualizeCurrentImageObjectInfos() {
    assert(cb_obj_infos_.size() == num_of_objects_per_image_);
    assert(model_filepaths_.size() == model_indices_.size());
    assert(num_of_objects_per_image_ <= model_indices_.size());
    renderer_->modelDrawers().clear();
    renderer_->bbxDrawers().clear();

    const ImageInfo& img_info = image_dataset_.image_infos.at(img_index_);
    const Eigen::Matrix3d K_inv = img_info.image_intrinsic.value().inverse();

    std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> sat_dist(0.95f, 1.0f);
    std::uniform_real_distribution<float> val_dist(0.95f, 1.0f);

    for (const auto& obj_info : cb_obj_infos_) {
      MeshData mesh = loadRandomMesh();

      const float golden_ratio_conjugate = 0.618033988749895f;
      const float hue = 360.0f * std::fmod(hue_dist(rnd_eng_) + golden_ratio_conjugate, 1.0f);
      const MeshData::ColorType color = CuteGL::makeRGBAfromHSV(hue, sat_dist(rnd_eng_), val_dist(rnd_eng_));
      CuteGL::colorizeMesh(mesh, color);


      const Eigen::Vector3d half_dimension = obj_info.dimension.value() / 2;
      const Eigen::AlignedBox3d bbx(-half_dimension, half_dimension);
      Eigen::Isometry3d pose = pose_from_obj_info(obj_info, K_inv);


      Eigen::Affine3d affine_model_pose = pose * Eigen::UniformScaling<double>(bbx.diagonal().norm());


      renderer_->modelDrawers().addItem(affine_model_pose.cast<float>(), mesh);
      renderer_->bbxDrawers().addItem(pose.cast<float>(), bbx.cast<float>());
    }
  }

  MeshData loadRandomMesh() {
    ++model_index_;
    if (model_index_ >= model_indices_.size()) {
      model_index_ = 0;
      std::shuffle(model_indices_.begin(), model_indices_.end(), rnd_eng_);
    }

    const std::string model_file = model_filepaths_.at(model_indices_.at(model_index_));
    return loadMeshFromPLY(model_file);
  }


  virtual void handleKeyboardAction(int action) {
    switch (action) {
      case NEXT_IMAGE: {
        img_index_ = std::min(image_dataset_.image_infos.size() - 1, img_index_ + 1);
        update();
        break;
      }
      case PREV_IMAGE: {
        img_index_ = std::max(0, int(img_index_) - 1);
        update();
        break;
      }
      default:
        WindowRenderViewer::handleKeyboardAction(action);
        break;
    }
  }

  MultiObjectRenderer* renderer_;
  std::size_t num_of_objects_per_image_;

  ImageDataset image_dataset_;
  ImageInfo::ImageObjectInfos all_object_infos_;
  std::size_t img_index_;

  std::vector<std::string> model_filepaths_;
  std::vector<std::size_t> model_indices_;
  std::size_t model_index_;

  ImageInfo::ImageObjectInfos cb_obj_infos_;

  std::mt19937 rnd_eng_;
};

}  // namespace CuteGL


int main(int argc, char **argv) {

  QApplication app(argc, argv);

  using namespace CuteGL;
  using namespace Eigen;

  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());
  renderer->setDisplayAxis(false);

  const std::string kitti_gt_json_file = RENDERFOR3DATA_ROOT_DIR "/data/kitti_trainval_full.json";
  ViewpointBrowser viewer(renderer.get(), 32, kitti_gt_json_file);
  viewer.setBackgroundColor(0, 0, 0);

  const int W = 1600;
  const int H = 800;

  viewer.resize(W, H);
  viewer.setSceneRadius(100.0f);

  viewer.camera().extrinsics() = Isometry3f::Identity();
  viewer.showAndWaitTillExposed();

  renderer->phongShader().program.bind();
  renderer->phongShader().setLightPosition(0.0f, -50.0f, 10.0f);
  renderer->phongShader().program.release();

  std::cout << "Reading model filelist ..." << std::flush;
  std::size_t num_of_models = viewer.readModelFilesList(RENDERFOR3DATA_ROOT_DIR "/data/ShapeNetCore_v1_clean_cars.txt",
                                                        RENDERFOR3DATA_ROOT_DIR "/data/ShapeNetCore_v1_PLY/Cars/", ".ply");
  std::cout << "We now have " << num_of_models << " models." << std::endl;

  viewer.update();
  app.exec();
}
