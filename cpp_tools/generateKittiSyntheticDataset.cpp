/**
 * @file generateKittiSyntheticDataset.cpp
 * @brief generateKittiSyntheticDataset
 *
 * @author Abhijit Kundu
 */

#include "RenderFor3Data/Config.h"
#include "RenderFor3Data/ImageDataset.h"
#include <CuteGL/Renderer/MultiObjectRenderer.h>
#include <CuteGL/Surface/OffScreenRenderViewer.h>
#include <CuteGL/Core/Config.h>
#include <CuteGL/IO/ImportPLY.h>
#include <CuteGL/Core/PoseUtils.h>
#include <CuteGL/Core/MeshUtils.h>
#include <CuteGL/Utils/ColorUtils.h>
#include <CuteGL/Geometry/ComputeAlignedBox.h>
#include <CuteGL/Geometry/OrientedBoxHelper.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/EulerAngles>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/progress.hpp>
#include <QGuiApplication>
#include <unordered_set>
#include <random>
#include <fstream>
#include <iostream>

namespace fs = boost::filesystem;

// TODO: In newer boost or c++17 we have relative
fs::path relative_path(fs::path to, fs::path from) {
   // Start at the root path and while they are the same then do nothing then when they first
   // diverge take the remainder of the two path and replace the entire from path with ".."
   // segments.
   fs::path::const_iterator fromIter = from.begin();
   fs::path::const_iterator toIter = to.begin();

   // Loop through both
   while (fromIter != from.end() && toIter != to.end() && (*toIter) == (*fromIter))
   {
      ++toIter;
      ++fromIter;
   }

   fs::path finalPath;
   while (fromIter != from.end())
   {
      finalPath /= "..";
      ++fromIter;
   }

   while (toIter != to.end())
   {
      finalPath /= *toIter;
      ++toIter;
   }

   return finalPath;
}

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

class MultiObjectDatasetGenerator {
 public:
  using MeshType = CuteGL::Mesh<float, float, unsigned char, int>;

  MultiObjectDatasetGenerator(const std::string& dataset_name,
                              const std::size_t num_of_objects_per_image,
                              const std::string& kitti_gt_json_file)
      : renderer_(new CuteGL::MultiObjectRenderer()),
        viewer_(renderer_.get()),
        num_of_objects_per_image_(num_of_objects_per_image),
        img_index_(0),
        rnd_eng_ {42} {

    real_image_dataset_ = loadImageDatasetFromJson(kitti_gt_json_file);
    for (const auto& image_info : real_image_dataset_.image_infos) {
      for (const auto& obj_info : image_info.object_infos.value()) {
        all_object_infos_.push_back(obj_info);
      }
    }

    synth_image_dataset_.name = dataset_name;
    synth_image_dataset_.rootdir = fs::path(RENDERFOR3DATA_ROOT_DIR) / fs::path("data");
    assert(fs::exists(synth_image_dataset_.rootdir));

    fs::path image_fp = synth_image_dataset_.rootdir / dataset_name / "color_gl" / "%s_color.png";
    fs::path segm_fp = synth_image_dataset_.rootdir / dataset_name / "segm_gl" / "%s_segm.png";

    if (!fs::exists(image_fp.parent_path()))
      fs::create_directories(image_fp.parent_path());

    if (!fs::exists(segm_fp.parent_path()))
      fs::create_directories(segm_fp.parent_path());

    image_file_fmt_ = boost::format(image_fp.string());
    segm_file_fmt_ = boost::format(segm_fp.string());

    viewer_.setBackgroundColor(0, 0, 0, 0);
    renderer_->setDisplayAxis(false);
    renderer_->setDisplayGrid(false);

    {
      const ImageInfo& img_info = real_image_dataset_.image_infos.at(0);
      const Eigen::Vector2i image_size = img_info.image_size.value();
      const Eigen::Matrix3f K = img_info.image_intrinsic.value().cast<float>();

      viewer_.resize(image_size.x(), image_size.y());
      viewer_.camera().intrinsics() = CuteGL::getGLPerspectiveProjection(K, image_size.x(), image_size.y(), 0.1f, 100.0f);
      viewer_.camera().extrinsics() = Eigen::Isometry3f::Identity();
    }

    viewer_.create();
    viewer_.makeCurrent();

    renderer_->phongShader().program.bind();
    renderer_->phongShader().setLightPosition(0.0f, -50.0f, 10.0f);
    renderer_->phongShader().program.release();
  }

  CuteGL::OffScreenRenderViewer& viewer() {return viewer_;}
  const CuteGL::OffScreenRenderViewer& viewer() const {return viewer_;}

  void loadAllModels(const std::string& models_list_filepath,
                            const std::string& prefix,
                            const std::string& suffix = "") {
    {
      std::ifstream file(models_list_filepath.c_str(), std::ios::in);
      if (!file.is_open()) {
        throw std::runtime_error("Cannot open File from " + models_list_filepath);
      }

      std::string line;
      while (std::getline(file, line)) {
        const std::string model_full_filepath = prefix + line + suffix;
        model_relative_filepaths_.push_back(relative_path(model_full_filepath, synth_image_dataset_.rootdir).string());
      }
      file.close();
    }

    const std::size_t num_of_models = model_relative_filepaths_.size();

    std::vector<Eigen::Matrix<unsigned char, 1, 4> > mesh_colors(num_of_models);
    {
      std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
      std::uniform_real_distribution<float> sat_dist(0.95f, 1.0f);
      std::uniform_real_distribution<float> val_dist(0.95f, 1.0f);
      for (std::size_t i = 0; i < num_of_models; ++i) {
        const float golden_ratio_conjugate = 0.618033988749895f;
        const float hue = 360.0f * std::fmod(hue_dist(rnd_eng_) + golden_ratio_conjugate, 1.0f);
        mesh_colors[i] = CuteGL::makeRGBAfromHSV(hue, sat_dist(rnd_eng_), val_dist(rnd_eng_)).transpose();
      }
    }

    {
      // Load all models
      std::cout << "Loading all " << num_of_models << " models ..." << std::endl;

      boost::progress_display show_progress(num_of_models);

      models_.resize(num_of_models);
      model_dimensions_.resize(num_of_models);

#pragma omp parallel for
      for (std::size_t i = 0; i < num_of_models; ++i) {
        CuteGL::MeshData legacy_mesh = CuteGL::loadMeshFromPLY((synth_image_dataset_.rootdir / model_relative_filepaths_.at(i)).string());
        Eigen::AlignedBox3f bbx = CuteGL::computeAlignedBox(legacy_mesh);
        model_dimensions_[i] = bbx.sizes().cast<double>();
        MeshType& mesh = models_[i];
        {
          mesh.positions.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
          mesh.normals.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
          for (Eigen::Index vid = 0; vid < mesh.positions.rows(); ++vid) {
            mesh.positions.row(vid) = legacy_mesh.vertices[vid].position;
            mesh.normals.row(vid) = legacy_mesh.vertices[vid].normal;
          }
          mesh.colors.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
          mesh.colors.rowwise() = mesh_colors[i];
          mesh.labels.setConstant(legacy_mesh.vertices.size(), i);
          mesh.faces = legacy_mesh.faces;
        }

        ++show_progress;
      }
    }

    assert(models_.size() == model_relative_filepaths_.size());
    assert(model_dimensions_.size() == model_relative_filepaths_.size());
  }

  void renderAndGenerateDataset() {
    std::cout << "Rendering and generating dataset ..." << std::endl;
    boost::progress_display show_progress(real_image_dataset_.image_infos.size());
    for (std::size_t i = 0; i < real_image_dataset_.image_infos.size(); ++i) {
      img_index_ = i;
      sampleCurrentImageObjectInfos();
      renderAndCheck();
      ++show_progress;
    }
  }

  void save_dataset(const std::string& filename) {
    saveImageDatasetToJson(synth_image_dataset_, filename);
  }

 protected:
  void setCameraIntrinsics(const Eigen::Matrix3d& K, const Eigen::Vector2i& image_size) {
    viewer_.resize(image_size[0], image_size[1]);
    viewer_.camera().intrinsics() = CuteGL::getGLPerspectiveProjection(K.cast<float>(), image_size[0], image_size[1], 0.1f, 100.0f);
    viewer_.recreateFBO();
  }

  void sampleCurrentImageObjectInfos() {
    cb_obj_infos_.clear();

    const ImageInfo& img_info = real_image_dataset_.image_infos.at(img_index_);

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
        vp = vp.unaryExpr(CuteGL::WrapToPi());

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
    obj_info.shape_file = model_relative_filepaths_.at(best_model_id);
  }

  void renderAndCheck() {
    if (num_of_objects_per_image_ != cb_obj_infos_.size())
      throw std::runtime_error("num_of_objects_per_image_ != cb_obj_infos_.size()");

    const ImageInfo& real_img_info = real_image_dataset_.image_infos.at(img_index_);

    const int H = viewer_.height();
    const int W = viewer_.width();
    assert(real_img_info.image_size.value() == Eigen::Vector2i(W, H));

    ImageInfo synth_img_info;
    synth_img_info.image_intrinsic = real_img_info.image_intrinsic.value();
    synth_img_info.image_size = Eigen::Vector2i(W, H);


    assert(cb_obj_infos_.size() == num_of_objects_per_image_);
    renderer_->modelDrawers().clear();
    renderer_->bbxDrawers().clear();

    const Eigen::Matrix3d K = synth_img_info.image_intrinsic.value();
    const Eigen::Matrix3d K_inv = K.inverse();


    for (size_t i = 0; i < num_of_objects_per_image_; ++i) {
      ImageObjectInfo& obj_info = cb_obj_infos_.at(i);

      std::size_t model_id = obj_info.id.value();
      MeshType& mesh = models_.at(model_id);
      mesh.labels.setConstant(i + 1);

      const Eigen::Vector3d half_dimension = obj_info.dimension.value() / 2;
      const Eigen::AlignedBox3d bbx(-half_dimension, half_dimension);
      Eigen::Isometry3d pose = pose_from_obj_info(obj_info, K_inv);

      Eigen::Affine3d affine_model_pose = pose * Eigen::UniformScaling<double>(bbx.diagonal().norm());
      renderer_->modelDrawers().addItem(affine_model_pose.cast<float>(), mesh);

      {
        // Compute bbx_amodal by projecting the vertices
        Eigen::Matrix2Xd img_projs = (K * affine_model_pose * mesh.positions.transpose().cast<double>()).colwise().hnormalized();
        Eigen::Vector4d bbx_amodal(img_projs.row(0).minCoeff(), img_projs.row(1).minCoeff(),
                                   img_projs.row(0).maxCoeff(), img_projs.row(1).maxCoeff());

        Eigen::Vector4d bbx_truncated(std::min(std::max(0.0, bbx_amodal[0]), double(W)),
                                      std::min(std::max(0.0, bbx_amodal[1]), double(H)),
                                      std::min(std::max(0.0, bbx_amodal[2]), double(W)),
                                      std::min(std::max(0.0, bbx_amodal[3]), double(H)));

        obj_info.bbx_amodal = bbx_amodal;

        double bbx_amodal_area = (bbx_amodal[2] - bbx_amodal[0]) * (bbx_amodal[3] - bbx_amodal[1]);
        double bbx_truncated_area = (bbx_truncated[2] - bbx_truncated[0]) * (bbx_truncated[3] - bbx_truncated[1]);
        assert(bbx_truncated_area <= bbx_amodal_area);
        assert(bbx_amodal_area > 0.0);

        obj_info.truncation = 1.0 - (bbx_truncated_area / bbx_amodal_area);
      }
    }

    assert(renderer_->modelDrawers().size() == num_of_objects_per_image_);

    // Render
    viewer_.render();


    const std::string image_name = fs::path(real_img_info.image_file.value()).stem().string();

    {
      // Save color image
      QImage color_image = viewer_.readColorBuffer();
      std::string image_file_path = boost::str(boost::format(image_file_fmt_) % image_name);
      color_image.save(QString::fromStdString(image_file_path));
      synth_img_info.image_file = relative_path(image_file_path, synth_image_dataset_.rootdir).string();
      assert(color_image.width() == W);
      assert(color_image.height() == H);
    }

    using Image32FC1 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Image8UC1 = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Image8UC1 label_image(H, W);
    {

      Image32FC1 image_32FC1(H, W);
      viewer_.readLabelBuffer(image_32FC1.data());
      image_32FC1.colwise().reverseInPlace();
      label_image = image_32FC1.cast<unsigned char>();
      cv::Mat cv_image(H, W, CV_8UC1, label_image.data());
      std::string segm_file_path = boost::str(boost::format(segm_file_fmt_) % image_name);
      cv::imwrite(segm_file_path, cv_image);
      synth_img_info.segm_file = relative_path(segm_file_path, synth_image_dataset_.rootdir).string();
    }

    // Compute 2d visible boxes
    std::vector<Eigen::AlignedBox2i> visible_boxes(num_of_objects_per_image_);
    std::vector<int> visible_pixel_counts(num_of_objects_per_image_, 0);
    for (Eigen::Index y = 0; y < H; ++y)
      for (Eigen::Index x = 0; x < W; ++x) {
        unsigned char label = label_image(y, x);
        assert(label <= num_of_objects_per_image_);
        if (label > 0) {
          visible_boxes.at(label - 1).extend(Eigen::Vector2i(x, y));
          ++visible_pixel_counts.at(label - 1);
        }
      }

    for (size_t i = 0; i < num_of_objects_per_image_; ++i) {
      ImageObjectInfo& obj_info = cb_obj_infos_.at(i);
      const Eigen::AlignedBox2i& bbx_visible = visible_boxes.at(i);

      obj_info.bbx_visible = Eigen::Vector4d(bbx_visible.min().x(), bbx_visible.min().y(),
                                             bbx_visible.max().x() + 1, bbx_visible.max().y() + 1);

      {
        // We no longer need all the previous drawers (we need to start rendering on object at a time)
        renderer_->modelDrawers().clear();

        std::size_t model_id = obj_info.id.value();
        MeshType& mesh = models_.at(model_id);
        obj_info.id = i + 1;
        mesh.labels.setConstant(obj_info.id.value());

        const Eigen::Vector3d half_dimension = obj_info.dimension.value() / 2;
        const Eigen::AlignedBox3d bbx(-half_dimension, half_dimension);
        Eigen::Isometry3d pose = pose_from_obj_info(obj_info, K_inv);

        Eigen::Affine3d affine_model_pose = pose * Eigen::UniformScaling<double>(bbx.diagonal().norm());
        renderer_->modelDrawers().addItem(affine_model_pose.cast<float>(), mesh);

        viewer_.render();
        {
          Image32FC1 image_32FC1(H, W);
          viewer_.readLabelBuffer(image_32FC1.data());
          Eigen::Index amodal_pixel_count = (image_32FC1.array() > 0.1f).count();

          if (amodal_pixel_count)
            obj_info.occlusion = 1.0 - (visible_pixel_counts.at(i) / (double)amodal_pixel_count);
          else
            obj_info.occlusion = 1.0;

          assert (obj_info.occlusion.value() <= 1.0);
          assert (obj_info.occlusion.value() >= 0.0);
        }

      }
    }

    synth_img_info.object_infos = cb_obj_infos_;
    synth_image_dataset_.image_infos.push_back(synth_img_info);
  }



 private:
  std::unique_ptr<CuteGL::MultiObjectRenderer> renderer_;
  CuteGL::OffScreenRenderViewer viewer_;

  std::size_t num_of_objects_per_image_;

  ImageDataset synth_image_dataset_;
  ImageDataset real_image_dataset_;
  ImageInfo::ImageObjectInfos all_object_infos_;
  std::size_t img_index_;

  std::vector<MeshType> models_;
  std::vector<Eigen::Vector3d> model_dimensions_;
  std::vector<std::string> model_relative_filepaths_;

  ImageInfo::ImageObjectInfos cb_obj_infos_;

  std::mt19937 rnd_eng_;

  boost::format image_file_fmt_;
  boost::format segm_file_fmt_;
};

int main(int argc, char **argv) {

  QGuiApplication app(argc, argv);

  const std::string dataset_name = "SyntheticKITTI32";
  const std::size_t num_of_objects_per_image = 32;
  const std::string kitti_gt_json_file = RENDERFOR3DATA_ROOT_DIR "/data/kitti_trainval_full.json";

  MultiObjectDatasetGenerator dataset_generator(dataset_name, num_of_objects_per_image, kitti_gt_json_file);

  dataset_generator.loadAllModels(RENDERFOR3DATA_ROOT_DIR "/data/cars_shape_files_ply.txt",
                                  RENDERFOR3DATA_ROOT_DIR "/data/CityShapes/");

  dataset_generator.renderAndGenerateDataset();
//
  std::cout << "Saving dataset ..." << std::flush;
  dataset_generator.save_dataset(dataset_name + ".json");
  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}


