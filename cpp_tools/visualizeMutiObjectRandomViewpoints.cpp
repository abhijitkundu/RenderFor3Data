/**
 * @file visualizeMutiObjectRandomViewpoints.cpp
 * @brief visualizeMutiObjectRandomViewpoints
 *
 * @author Abhijit Kundu
 */

#include "RenderFor3Data/Config.h"
#include <CuteGL/Renderer/MultiObjectRenderer.h>
#include <CuteGL/Surface/WindowRenderViewer.h>
#include <CuteGL/Core/Config.h>
#include <CuteGL/IO/ImportPLY.h>
#include <CuteGL/Core/PoseUtils.h>
#include <CuteGL/Core/MeshUtils.h>
#include <CuteGL/Utils/ColorUtils.h>
#include <CuteGL/Geometry/ComputeAlignedBox.h>
#include <CuteGL/Geometry/OrientedBoxHelper.h>

#include <Eigen/EulerAngles>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
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
        rnd_eng_ {42},
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
    camera().intrinsics() = getGLPerspectiveProjection(fx, fy, 0.0f, cx, cy, img_width, img_height, near_z, far_z);
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
      vp = vp.unaryExpr(WrapToPi());

      viewpoints_.push_back(vp);
    }

    file.close();

    vp_indices_.resize(viewpoints_.size());
    std::iota(vp_indices_.begin(), vp_indices_.end(), 0);
    std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);

    return vp_indices_.size();
  }

  std::size_t readModelFilesList(const std::string& models_list_filepath, const std::string& prefix, const std::string& suffix="") {
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
    renderer_->bbxDrawers().clear();
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

      const std::string model_file = model_filepaths_.at(model_indices_.at(model_index_));

      auto mesh = loadMeshFromPLY(model_file);
      Eigen::AlignedBox3f bbx  = computeAlignedBox(mesh);
      model_bbx_sizes_.push_back(bbx.sizes());


      const float golden_ratio_conjugate = 0.618033988749895f;
      const float hue = 360.0f * std::fmod(hue_dist(rnd_eng_) + golden_ratio_conjugate, 1.0f);
      const MeshData::ColorType color = CuteGL::makeRGBAfromHSV(hue, sat_dist(rnd_eng_), val_dist(rnd_eng_));
      CuteGL::colorizeMesh(mesh, color);
      renderer_->modelDrawers().addItem(Eigen::Affine3f::Identity(), mesh);
      renderer_->bbxDrawers().addItem(Eigen::Affine3f::Identity(), bbx);
    }

    assert(model_filepaths_.size() == model_indices_.size());
  }

  void setModelPoses() {
    assert(viewpoints_.size() == vp_indices_.size());
    assert(renderer_->modelDrawers().poses().size() == renderer_->bbxDrawers().poses().size());
    const std::size_t number_of_objects = renderer_->modelDrawers().poses().size();


    std::uniform_real_distribution<double> x_dis(0.0, width());
    std::uniform_real_distribution<double> y_dis(0.0, height());
    std::uniform_real_distribution<double> z_dis(1.0, 29.0);

    MultiObjectRenderer::ModelDrawers::Poses& model_poses = renderer_->modelDrawers().poses();
    MultiObjectRenderer::BoundingBoxDrawers::Poses& bbx_poses = renderer_->bbxDrawers().poses();

    const Eigen::Matrix3f Kinv = K_.inverse();

    for (size_t i = 0; i < number_of_objects; ++i) {
      ++vp_index_;
      if (vp_index_ >= vp_indices_.size()) {
        vp_index_ = 0;
        std::shuffle(vp_indices_.begin(), vp_indices_.end(), rnd_eng_);
      }
      const Eigen::Vector3f& vp = viewpoints_[vp_indices_[vp_index_]];

      // Loop and keep sampling new center_proj until we find a collision free pose
      while (true) {
        float center_dist = z_dis(rnd_eng_);
        Eigen::Isometry3f vp_pose = getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), center_dist);
        Eigen::Vector2f center_proj(x_dis(rnd_eng_), y_dis(rnd_eng_));
        Eigen::Vector3f center_proj_ray = Kinv * center_proj.homogeneous();
        Eigen::Isometry3f pose = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), center_proj_ray) * vp_pose;

        bool collision_free = true;

        // check for all previous poses for collision
        for (size_t j = 0; j < i; ++j) {
          if (checkOrientedBoxCollision(pose, model_bbx_sizes_.at(i), model_poses.at(j), model_bbx_sizes_.at(j))) {
            collision_free = false;
            break;
          }
        }

        if (collision_free) {
          model_poses[i] = pose;
          bbx_poses[i] = pose;
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


  std::vector<Eigen::Vector3f> model_bbx_sizes_;
  std::mt19937 rnd_eng_;
  Eigen::Matrix3f K_;
};

}  // namespace CuteGL

int main(int argc, char **argv) {

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;

  po::options_description generic_options("Generic Options");
  generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
  config_options.add_options()("objects_per_image,n", po::value<std::size_t>()->default_value(32), "# objects per image")
                              ("viewpoint_file,v", po::value<fs::path>()->required(), "Path to viewpoint ditribution file")
                              ("shape_files_list,s", po::value<fs::path>()->required(), "Path to shape files list file")
                              ;

  po::options_description cmdline_options;
  cmdline_options.add(generic_options).add(config_options);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    po::notify(vm);
  } catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  if (vm.count("help")) {
    std::cout << cmdline_options << '\n';
    return EXIT_SUCCESS;
  }

  const fs::path viewpoint_file = vm["viewpoint_file"].as<fs::path>();
  const fs::path shape_files_list = vm["shape_files_list"].as<fs::path>();

  if (!fs::is_regular_file(viewpoint_file)) {
    std::cout << viewpoint_file << " does not exist (or not a regular file)\n";
    return EXIT_FAILURE;
  }

  if (!fs::is_regular_file(shape_files_list)) {
    std::cout << shape_files_list << " does not exist (or not a regular file)\n";
    return EXIT_FAILURE;
  }

  QApplication app(argc, argv);

  using namespace CuteGL;
  using namespace Eigen;

  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());
  renderer->setDisplayAxis(false);

  ViewpointBrowser viewer(renderer.get(), vm["objects_per_image"].as<std::size_t>());
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
  std::size_t num_of_vps = viewer.readViewpoints(viewpoint_file.string());
  std::cout << "We now have " << num_of_vps << " viewpoints." << std::endl;

  std::cout << "Reading model filelist ..." << std::flush;
  std::size_t num_of_models = viewer.readModelFilesList(shape_files_list.string(),
                                                        RENDERFOR3DATA_ROOT_DIR "/data/CityShapes/");
  std::cout << "We now have " << num_of_models << " models." << std::endl;

  app.exec();
}

