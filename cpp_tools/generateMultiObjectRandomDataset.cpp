/**
 * @file generateMultiObjectRandomDataset.cpp
 * @brief generateMultiObjectRandomDataset
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
#include <CuteGL/Core/ColorUtils.h>
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

class MultiObjectDatasetGenerator {
 public:
  using MeshType = CuteGL::Mesh<float, float, unsigned char, int>;

  MultiObjectDatasetGenerator(const std::size_t num_of_objects_per_image)
      : renderer_(new CuteGL::MultiObjectRenderer()),
        viewer_(renderer_.get()),
        num_of_objects_per_image_(num_of_objects_per_image),
        vp_index_(0),
        model_index_(0),
        rnd_eng_ {42},
        K_(Eigen::Matrix3d::Identity()) {

    viewer_.setBackgroundColor(0, 0, 0);
    renderer_->setDisplayAxis(false);

    //  const int W = 960;
    //  const int H = 540;
    //  const float focal_length = 1050.0f;

    const int W = 1600;
    const int H = 800;
    const double focal_length = 1760.0f;

    K_ << focal_length, 0.0, W / 2.0, 0.0, focal_length, H / 2.0, 0.0f, 0.0f, 1.0;

    viewer_.resize(W, H);
    viewer_.camera().intrinsics() = CuteGL::getGLPerspectiveProjection(K_.cast<float>(), W, H, 0.1f, 50.0f);
    viewer_.camera().extrinsics() = Eigen::Isometry3f::Identity();

    viewer_.create();
    viewer_.makeCurrent();

    renderer_->phongShader().program.bind();
    renderer_->phongShader().setLightPosition(0.0f, -50.0f, 10.0f);
    renderer_->phongShader().program.release();

    dataset_.name = "RandomFlyingCars";
    dataset_.rootdir = fs::path(RENDERFOR3DATA_ROOT_DIR) / fs::path("data");
    assert(fs::exists(dataset_.rootdir));

    image_file_fmt_ = boost::format((dataset_.rootdir / "RandomFlyingCars" / "color_gl"/ "%08i_color.png").string());
    segm_file_fmt_ = boost::format((dataset_.rootdir / "RandomFlyingCars" / "segm_gl"/ "%08i_segm.png").string());
  }

  CuteGL::OffScreenRenderViewer& viewer() {return viewer_;}
  const CuteGL::OffScreenRenderViewer& viewer() const {return viewer_;}

  std::size_t readViewpoints(const std::string& vp_distrib_filepath) {
    std::ifstream file(vp_distrib_filepath.c_str(), std::ios::in);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open File from " + vp_distrib_filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      Eigen::Vector3d vp;
      if (!(iss >> vp[0] >> vp[1] >> vp[2])) {
        break;
      }  // error

      vp *= M_PI / 180.0;
      vp = vp.unaryExpr(CuteGL::WrapToPi());

      viewpoints_.push_back(vp);
    }

    file.close();

    vp_indices_.resize(viewpoints_.size());
    std::iota(vp_indices_.begin(), vp_indices_.end(), 0);
    std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);

    return vp_indices_.size();
  }

  std::size_t readModelFilesList(const std::string& models_list_filepath, const std::string& prefix,
                                 const std::string& suffix) {
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

  void loadNextBatchOfModels() {
    assert(model_filepaths_.size() == model_indices_.size());
    assert(num_of_objects_per_image_ <= model_indices_.size());
    renderer_->modelDrawers().clear();
    cb_obj_infos_.clear();
    cb_meshes_.clear();

    std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> sat_dist(0.95f, 1.0f);
    std::uniform_real_distribution<float> val_dist(0.95f, 1.0f);


    for (std::size_t i = 0; i < num_of_objects_per_image_; ++i) {
      ++model_index_;
      if (model_index_ >= model_indices_.size()) {
        model_index_ = 0;
        std::shuffle(model_indices_.begin(), model_indices_.end(), rnd_eng_);
      }
      const std::string model_file = model_filepaths_.at(model_indices_.at(model_index_));

      ImageObjectInfo obj_info;
      obj_info.id = i + 1;
      obj_info.category = "car";
      obj_info.shape_file = relative_path(model_file, dataset_.rootdir).string();

      CuteGL::MeshData legacy_mesh = CuteGL::loadMeshFromPLY(model_file);
      Eigen::AlignedBox3f bbx  = computeAlignedBox(legacy_mesh);
      obj_info.dimension = bbx.sizes().cast<double>();

      const float golden_ratio_conjugate = 0.618033988749895f;
      const float hue = 360.0f * std::fmod(hue_dist(rnd_eng_) + golden_ratio_conjugate, 1.0f);
      const CuteGL::MeshData::ColorType color = CuteGL::makeRGBAfromHSV(hue, sat_dist(rnd_eng_), val_dist(rnd_eng_));


      MeshType mesh;
      {
        mesh.positions.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
        mesh.normals.resize(legacy_mesh.vertices.size(), Eigen::NoChange);

        for (Eigen::Index vid= 0; vid < mesh.positions.rows(); ++vid) {
          mesh.positions.row(vid) = legacy_mesh.vertices[vid].position;
          mesh.normals.row(vid) = legacy_mesh.vertices[vid].normal;
        }

        mesh.colors.resize(legacy_mesh.vertices.size(), Eigen::NoChange);
        mesh.colors.rowwise() = color.transpose();

        mesh.labels.setConstant(legacy_mesh.vertices.size(), obj_info.id.value());

        mesh.faces = legacy_mesh.faces;
      }


      renderer_->modelDrawers().addItem(Eigen::Affine3f::Identity(), mesh);

      cb_obj_infos_.push_back(obj_info);
      cb_meshes_.push_back(mesh);
    }

    assert(cb_meshes_.size() == num_of_objects_per_image_);
    assert(cb_obj_infos_.size() == num_of_objects_per_image_);
    assert(renderer_->modelDrawers().size() == num_of_objects_per_image_);
  }

  void generateModelPoses() {
    assert(viewpoints_.size() == vp_indices_.size());
    std::uniform_real_distribution<double> x_dis(0.0, viewer_.width());
    std::uniform_real_distribution<double> y_dis(0.0, viewer_.height());
    std::uniform_real_distribution<double> z_dis(1.0, 30.0);

    auto& poses = renderer_->modelDrawers().poses();
    const Eigen::Matrix3d Kinv = K_.inverse();
    const std::size_t number_of_objects = renderer_->modelDrawers().poses().size();
    assert(cb_obj_infos_.size() == number_of_objects);

    for (size_t i = 0; i < number_of_objects; ++i) {
      ImageObjectInfo& obj_info_i = cb_obj_infos_.at(i);

      ++vp_index_;
      if (vp_index_ >= vp_indices_.size()) {
        vp_index_ = 0;
        std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);
      }
      const Eigen::Vector3d& vp = viewpoints_[vp_indices_[vp_index_]];
      obj_info_i.viewpoint = vp;

      // Loop and keep sampling new center_proj until we find a collision free pose
      while (true) {
        double center_dist = z_dis(rnd_eng_);
        Eigen::Isometry3d vp_pose = CuteGL::getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), center_dist);
        Eigen::Vector2d center_proj(x_dis(rnd_eng_), y_dis(rnd_eng_));
        Eigen::Vector3d center_proj_ray = Kinv * center_proj.homogeneous();
        Eigen::Isometry3d pose = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), center_proj_ray) * vp_pose;

        bool collision_free = true;

        // check for all previous poses for collision
        for (size_t j = 0; j < i; ++j) {
          ImageObjectInfo& obj_info_j = cb_obj_infos_.at(j);
          if (CuteGL::checkOrientedBoxCollision(pose, obj_info_i.dimension.value(), poses.at(j).cast<double>(), obj_info_j.dimension.value())) {
            collision_free = false;
            break;
          }
        }

        if (collision_free) {
          poses[i] = pose.cast<float>();
          obj_info_i.center_proj = center_proj;
          obj_info_i.center_dist = center_dist;
          break;
        }
      }
    }
  }

  void render_and_check() {
    const std::size_t number_of_objects = renderer_->modelDrawers().poses().size();
    assert(cb_obj_infos_.size() == number_of_objects);
    assert(cb_meshes_.size() == number_of_objects);

    const int H = viewer_.height();
    const int W = viewer_.width();

    ImageInfo img_info;
    img_info.image_intrinsic = K_;
    img_info.image_size = Eigen::Vector2i(W, H);

    const Eigen::Matrix3d K_inv = K_.inverse();

    // Render and update bbx_visible
    viewer_.render();
    {
      // Save color image
      QImage color_image = viewer_.readColorBuffer();
      std::string image_file_path = boost::str(boost::format(image_file_fmt_) % dataset_.image_infos.size());
      color_image.save(QString::fromStdString(image_file_path));
      img_info.image_file = relative_path(image_file_path, dataset_.rootdir).string();
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
      std::string segm_file_path = boost::str(boost::format(segm_file_fmt_) % dataset_.image_infos.size());
      cv::imwrite(segm_file_path, cv_image);
      img_info.segm_file = relative_path(segm_file_path, dataset_.rootdir).string();
    }

    for (size_t i = 0; i < number_of_objects; ++i) {
      ImageObjectInfo& obj_info = cb_obj_infos_.at(i);
      const Eigen::Vector3d& vp = obj_info.viewpoint.value();
      Eigen::Isometry3d vp_pose = CuteGL::getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), obj_info.center_dist.value());
      Eigen::Vector3d center_proj_ray = K_inv * obj_info.center_proj.value().homogeneous();
      Eigen::Isometry3d pose = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), center_proj_ray) * vp_pose;

      // Compute bbx_amodal by projecting the vertices
      Eigen::Matrix2Xd img_projs = (img_info.image_intrinsic.value() * pose * cb_meshes_.at(i).positions.transpose().cast<double>()).colwise().hnormalized();
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

      obj_info.truncation = 1.0 - (bbx_truncated_area / bbx_amodal_area);
    }

    img_info.object_infos = cb_obj_infos_;
    dataset_.image_infos.push_back(img_info);
  }

  void save_dataset(const std::string& filename) {
    saveImageDatasetToJson(dataset_, filename);
  }

 private:
  std::unique_ptr<CuteGL::MultiObjectRenderer> renderer_;
  CuteGL::OffScreenRenderViewer viewer_;

  std::size_t num_of_objects_per_image_;

  std::vector<Eigen::Vector3d> viewpoints_;
  std::vector<std::size_t> vp_indices_;
  std::size_t vp_index_;

  std::vector<std::string> model_filepaths_;
  std::vector<std::size_t> model_indices_;
  std::size_t model_index_;

  std::mt19937 rnd_eng_;
  Eigen::Matrix3d K_;

  std::vector<MeshType> cb_meshes_;
  ImageInfo::ImageObjectInfos cb_obj_infos_;
  ImageDataset dataset_;

  boost::format image_file_fmt_;
  boost::format segm_file_fmt_;
};

int main(int argc, char **argv) {

  QGuiApplication app(argc, argv);

  const std::size_t num_of_objects_per_image = 32;
  MultiObjectDatasetGenerator dataset_generator(num_of_objects_per_image);

  std::cout << "Reading viewpoints ..." << std::flush;
  std::size_t num_of_vps = dataset_generator.readViewpoints(
      RENDERFOR3DATA_ROOT_DIR "/data/view_distribution/voc2012_kitti/car.txt");
  std::cout << "We now have " << num_of_vps << " viewpoints." << std::endl;

  std::cout << "Reading model filelist ..." << std::flush;
  std::size_t num_of_models = dataset_generator.readModelFilesList(
      RENDERFOR3DATA_ROOT_DIR "/data/ShapeNetCore_v1_clean_cars.txt",
      RENDERFOR3DATA_ROOT_DIR "/data/ShapeNetCore_v1_PLY/Cars/",".ply");
  std::cout << "We now have " << num_of_models << " models." << std::endl;

  std::cout << "Rendering Images ..." << std::endl;

  const int num_of_images_to_generate = 100;
  boost::progress_display show_progress(num_of_images_to_generate);
  for (int i = 0; i<num_of_images_to_generate; ++i ) {
    dataset_generator.loadNextBatchOfModels();
    dataset_generator.generateModelPoses();
    dataset_generator.render_and_check();
    ++show_progress;
  }

  std::cout << "Saving dataset ..." << std::flush;
  dataset_generator.save_dataset("temp.json");
  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}

