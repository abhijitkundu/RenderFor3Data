/**
 * @file generateMultiObjectRandomDataset.cpp
 * @brief generateMultiObjectRandomDataset
 *
 * @author Abhijit Kundu
 */

#include "RenderFor3Data/Config.h"
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

class MultiObjectDatasetGenerator {
 public:
  MultiObjectDatasetGenerator(const std::size_t num_of_objects_per_image)
      : renderer_(new CuteGL::MultiObjectRenderer()),
        viewer_(renderer_.get()),
        num_of_objects_per_image_(num_of_objects_per_image),
        vp_index_(0),
        model_index_(0),
        rnd_eng_ { std::random_device { }() },
        K_(Eigen::Matrix3f::Identity()) {

    viewer_.setBackgroundColor(0, 0, 0);
    renderer_->setDisplayAxis(false);

    //  const int W = 960;
    //  const int H = 540;
    //  const float focal_length = 1050.0f;

    const int W = 1600;
    const int H = 800;
    const float focal_length = 1760.0f;

    K_ << focal_length, 0.0f, W / 2.0f, 0.0f, focal_length, H / 2.0f, 0.0f, 0.0f, 1.0f;

    viewer_.resize(W, H);
    viewer_.camera().intrinsics() = CuteGL::getGLPerspectiveProjection(K_, W, H, 0.1f, 50.0f);
    viewer_.camera().extrinsics() = Eigen::Isometry3f::Identity();

    viewer_.create();
    viewer_.makeCurrent();

    renderer_->phongShader().program.bind();
    renderer_->phongShader().setLightPosition(0.0f, -50.0f, 10.0f);
    renderer_->phongShader().program.release();
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
    model_bbx_sizes_.clear();

    std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> sat_dist(0.95f, 1.0f);
    std::uniform_real_distribution<float> val_dist(0.95f, 1.0f);


    for (std::size_t i = 0; i < num_of_objects_per_image_; ++i) {
      ++model_index_;
      if (model_index_ >= model_indices_.size()) {
        model_index_ = 0;
        std::shuffle(model_indices_.begin(), model_indices_.end(), rnd_eng_);
      }

      CuteGL::MeshData legacy_mesh = CuteGL::loadMeshFromPLY(model_filepaths_.at(model_indices_.at(model_index_)));
      Eigen::AlignedBox3f bbx  = computeAlignedBox(legacy_mesh);
      model_bbx_sizes_.push_back(bbx.sizes());

      const float golden_ratio_conjugate = 0.618033988749895f;
      const float hue = 360.0f * std::fmod(hue_dist(rnd_eng_) + golden_ratio_conjugate, 1.0f);
      const CuteGL::MeshData::ColorType color = CuteGL::makeRGBAfromHSV(hue, sat_dist(rnd_eng_), val_dist(rnd_eng_));

      using MeshType = CuteGL::Mesh<float, float, unsigned char, int>;
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

        mesh.labels.setConstant(legacy_mesh.vertices.size(), i+1);

        mesh.faces = legacy_mesh.faces;
      }


      renderer_->modelDrawers().addItem(Eigen::Affine3f::Identity(), mesh);
    }

    assert(model_filepaths_.size() == model_indices_.size());
  }

  void generateModelPoses() {
    assert(viewpoints_.size() == vp_indices_.size());
    std::uniform_real_distribution<float> x_dis(0.0f, viewer_.width());
    std::uniform_real_distribution<float> y_dis(0.0f, viewer_.height());
    std::uniform_real_distribution<float> z_dis(1.0f, 30.0f);

    auto& poses = renderer_->modelDrawers().poses();
    const Eigen::Matrix3f Kinv = K_.inverse();
    const std::size_t number_of_objects = renderer_->modelDrawers().poses().size();
    assert(model_bbx_sizes_.size() == number_of_objects);

    for (size_t i = 0; i < number_of_objects; ++i) {
      ++vp_index_;
      if (vp_index_ >= vp_indices_.size()) {
        vp_index_ = 0;
        std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);
      }
      const Eigen::Vector3f& vp = viewpoints_[vp_indices_[vp_index_]];

      // Loop and keep sampling new center_proj until we find a collision free pose
      while (true) {
        Eigen::Isometry3f vp_pose = CuteGL::getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), z_dis(rnd_eng_));
        Eigen::Vector3f center_proj_ray = Kinv * Eigen::Vector3f(x_dis(rnd_eng_), y_dis(rnd_eng_), 1.0f);
        Eigen::Isometry3f pose = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), center_proj_ray) * vp_pose;

        bool collision_free = true;

        // check for all previous poses for collision
        for (size_t j = 0; j < i; ++j) {
          if (CuteGL::checkOrientedBoxCollision(pose, model_bbx_sizes_.at(i), poses.at(j), model_bbx_sizes_.at(j))) {
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

 private:
  std::unique_ptr<CuteGL::MultiObjectRenderer> renderer_;
  CuteGL::OffScreenRenderViewer viewer_;

  std::size_t num_of_objects_per_image_;

  std::vector<Eigen::Vector3f> viewpoints_;
  std::vector<std::size_t> vp_indices_;
  std::size_t vp_index_;

  std::vector<std::string> model_filepaths_;
  std::vector<std::size_t> model_indices_;
  std::size_t model_index_;

  std::vector<Eigen::Vector3f> model_bbx_sizes_;
  std::mt19937 rnd_eng_;
  Eigen::Matrix3f K_;
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

  const int H = dataset_generator.viewer().height();
  const int W = dataset_generator.viewer().width();

  using Image32FC1 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Image16SC1 = Eigen::Matrix<short int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Image32FC1 image_32FC1(H, W);
  Image16SC1 image_16SC1(H, W);

  const boost::format image_out_file_fmt("test_%i.png");

  const int num_of_images_to_generate = 100;
  boost::progress_display show_progress(num_of_images_to_generate);
  for (int i = 0; i<num_of_images_to_generate; ++i ) {
    dataset_generator.loadNextBatchOfModels();
    dataset_generator.generateModelPoses();
    dataset_generator.viewer().render();

    dataset_generator.viewer().readLabelBuffer(image_32FC1.data());
    image_32FC1.colwise().reverseInPlace();
    image_16SC1 = image_32FC1.cast<short int>();

    cv::Mat cv_image(H, W, CV_16SC1, image_16SC1.data());
    cv::imwrite(boost::str(boost::format(image_out_file_fmt) % i), cv_image);
    ++show_progress;
  }

  return EXIT_SUCCESS;
}

