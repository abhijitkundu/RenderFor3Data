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
#include <unordered_set>
#include <fstream>
#include <iostream>

Eigen::Isometry3d pose_from_obj_info(const ImageObjectInfo& obj_info, const Eigen::Matrix3d& K_inv) {
  Eigen::Vector3d vp = obj_info.viewpoint.value();
  Eigen::Isometry3d vp_pose = CuteGL::getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), obj_info.center_dist.value());
  Eigen::Vector3d center_proj_ray = K_inv * obj_info.center_proj.value().homogeneous();
  Eigen::Isometry3d pose = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), center_proj_ray) * vp_pose;
  return pose;
}

std::vector<std::size_t> chooseRandomSunset(std::size_t N, std::size_t k, std::mt19937& gen) {
  std::uniform_int_distribution<std::size_t> dis(0, N-1);
  std::unordered_set<std::size_t> set;
  while (set.size() < k) {
    set.insert(dis(gen));
  }
  std::vector<std::size_t> result(set.begin(), set.end());
  return result;
}

namespace CuteGL {

class ViewpointBrowser : public WindowRenderViewer {
 public:
  using MeshType = Mesh<float, float, unsigned char, int>;

  enum Action {
    NEXT_IMAGE = 100,
    PREV_IMAGE = 101,
  };

  explicit ViewpointBrowser(MultiObjectRenderer* renderer, const std::size_t num_of_objects_per_image, const std::string& kitti_gt_json_file)
      : WindowRenderViewer(renderer),
        renderer_(renderer),
        num_of_objects_per_image_(num_of_objects_per_image),
        img_index_(0),
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

  std::size_t loadAllModels(const std::string& models_list_filepath, const std::string& prefix, const std::string& suffix = "") {

    std::vector<std::string> model_filepaths;
    {
      std::ifstream file(models_list_filepath.c_str(), std::ios::in);
      if (!file.is_open()) {
        throw std::runtime_error("Cannot open File from " + models_list_filepath);
      }

      std::string line;
      while (std::getline(file, line)) {
        model_filepaths.push_back(prefix + line + suffix);
      }
      file.close();
    }

    {
      // Load all models
      models_.resize(model_filepaths.size());
      model_dimensions_.resize(model_filepaths.size());

      std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
      std::uniform_real_distribution<float> sat_dist(0.95f, 1.0f);
      std::uniform_real_distribution<float> val_dist(0.95f, 1.0f);

#pragma omp parallel for
      for (std::size_t i= 0; i < model_filepaths.size(); ++i) {
        MeshData legacy_mesh = loadMeshFromPLY(model_filepaths[i]);
        Eigen::AlignedBox3f bbx  = computeAlignedBox(legacy_mesh);
        model_dimensions_[i] = bbx.sizes().cast<double>();

        const float golden_ratio_conjugate = 0.618033988749895f;
        const float hue = 360.0f * std::fmod(hue_dist(rnd_eng_) + golden_ratio_conjugate, 1.0f);
        const CuteGL::MeshData::ColorType color = CuteGL::makeRGBAfromHSV(hue, sat_dist(rnd_eng_), val_dist(rnd_eng_));

        MeshType& mesh = models_[i];
        {
          mesh.positions.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
          mesh.normals.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
          for (Eigen::Index vid= 0; vid < mesh.positions.rows(); ++vid) {
            mesh.positions.row(vid) = legacy_mesh.vertices[vid].position;
            mesh.normals.row(vid) = legacy_mesh.vertices[vid].normal;
          }
          mesh.colors.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
          mesh.colors.rowwise() = color.transpose();
          mesh.labels.setConstant(legacy_mesh.vertices.size(), i);
          mesh.faces = legacy_mesh.faces;
        }
      }
    }

    assert(models_.size() == model_filepaths.size());
    assert(model_dimensions_.size() == model_filepaths.size());

    return model_filepaths.size();
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
      ImageObjectInfo obj_info = current_image_obj_infos[i];

      find_best_fitting_model(obj_info, 100);
      cb_obj_infos_.push_back(obj_info);
      cb_poses.push_back(pose_from_obj_info(obj_info, K_inv));
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

        find_best_fitting_model(random_obj_info, 10);
        const Eigen::Isometry3d pose = pose_from_obj_info(random_obj_info, K_inv);

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

  void find_best_fitting_model(ImageObjectInfo& obj_info, std::size_t K) {
    const Eigen::Vector3d gt_dimension_normalized = obj_info.dimension.value().normalized();
    std::vector<std::size_t> model_subset_indices = chooseRandomSunset(models_.size(), K, rnd_eng_);
    // Choose the best fitting model
    std::size_t best_model_id = 0;
    {
      double min_diff = std::numeric_limits<double>::max();
      for (std::size_t model_id : model_subset_indices) {
        double diff = (gt_dimension_normalized - model_dimensions_.at(model_id)).norm();
        if (diff < min_diff) {
          min_diff = diff;
          best_model_id = model_id;
        }
      }
    }
    obj_info.id = best_model_id;
    obj_info.dimension = model_dimensions_.at(best_model_id) * obj_info.dimension.value().norm();
  }


  void visualizeCurrentImageObjectInfos() {
    assert(cb_obj_infos_.size() == num_of_objects_per_image_);
    renderer_->modelDrawers().clear();
    renderer_->bbxDrawers().clear();

    const ImageInfo& img_info = image_dataset_.image_infos.at(img_index_);
    const Eigen::Matrix3d K_inv = img_info.image_intrinsic.value().inverse();

    for (const auto& obj_info : cb_obj_infos_) {
      std::size_t model_id = obj_info.id.value();

      const Eigen::Vector3d half_dimension = obj_info.dimension.value() / 2;
      const Eigen::AlignedBox3d bbx(-half_dimension, half_dimension);
      Eigen::Isometry3d pose = pose_from_obj_info(obj_info, K_inv);


      Eigen::Affine3d affine_model_pose = pose * Eigen::UniformScaling<double>(bbx.diagonal().norm());


      renderer_->modelDrawers().addItem(affine_model_pose.cast<float>(), models_.at(model_id));
      renderer_->bbxDrawers().addItem(pose.cast<float>(), bbx.cast<float>());
    }
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

  std::vector<MeshType> models_;
  std::vector<Eigen::Vector3d> model_dimensions_;

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

  std::cout << "Loading all models ..." << std::flush;
  std::size_t num_of_models = viewer.loadAllModels(RENDERFOR3DATA_ROOT_DIR "/data/cars_shape_files_ply.txt",
                                                   RENDERFOR3DATA_ROOT_DIR "/data/CityShapes/");
  std::cout << "We now have " << num_of_models << " models." << std::endl;

  viewer.update();
  app.exec();
}
