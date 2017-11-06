/**
 * @file renderSingleImageInfo.cpp
 * @brief renderSingleImageInfo
 *
 * @author Abhijit Kundu
 */

#include "RenderFor3Data/Config.h"
#include "RenderFor3Data/ImageDataset.h"
#include <CuteGL/Renderer/MultiObjectRenderer.h>
#include <CuteGL/Surface/OffScreenRenderViewer.h>
#include <CuteGL/IO/ImportViaAssimp.h>
#include <CuteGL/IO/ImportPLY.h>
#include <CuteGL/Geometry/ComputeAlignedBox.h>
#include <CuteGL/Geometry/OrientedBoxHelper.h>
#include <CuteGL/Core/PoseUtils.h>
#include <CuteGL/Core/MeshUtils.h>
#include <QGuiApplication>
#include <boost/filesystem.hpp>
#include <boost/filesystem/convenience.hpp>
#include <iostream>

Eigen::Isometry3d pose_from_obj_info(const ImageObjectInfo& obj_info, const Eigen::Matrix3d& K_inv) {
  Eigen::Vector3d vp = obj_info.viewpoint.value();
  Eigen::Isometry3d vp_pose = CuteGL::getExtrinsicsFromViewPoint(vp.x(), vp.y(), vp.z(), obj_info.center_dist.value());
  Eigen::Vector3d center_proj_ray = K_inv * obj_info.center_proj.value().homogeneous();
  Eigen::Isometry3d pose = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), center_proj_ray) * vp_pose;
  return pose;
}

int main(int argc, char **argv) {
  namespace fs = boost::filesystem;
  using namespace CuteGL;
  using namespace Eigen;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <path/to/single_image_info.json>\n";
    return EXIT_FAILURE;
  }

  fs::path image_info_file(argv[1]);
  if (!fs::exists(image_info_file)) {
    std::cout << image_info_file << " do not exist\n";
    return EXIT_FAILURE;
  }

  const ImageInfo img_info = loadImageInfoFromJson(image_info_file.string());

  QGuiApplication app(argc, argv);

  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());
  renderer->setDisplayAxis(false);
  renderer->setDisplayGrid(false);

  OffScreenRenderViewer viewer(renderer.get());

  viewer.setBackgroundColor(0, 0, 0, 0);

  const Matrix3d K = img_info.image_intrinsic.value();
  const Vector2i image_size = img_info.image_size.value();

  std::cout << "image_size = " << image_size.x() << "x" << image_size.y() << std::endl;
  std::cout << "K=\n" << K << std::endl;


  const Matrix3d K_inv = K.inverse();

  viewer.resize(image_size[0], image_size[1]);

  viewer.create();
  viewer.makeCurrent();


  viewer.camera().intrinsics() = getGLPerspectiveProjection(K.cast<float>(), image_size[0], image_size[1], 0.1f, 100.0f);
  viewer.camera().extrinsics() = Eigen::Isometry3f::Identity();

  renderer->phongShader().program.bind();
  renderer->phongShader().setLightPosition(0.0f, -50.0f, 10.0f);
  renderer->phongShader().program.release();

  fs::path root_dir(RENDERFOR3DATA_ROOT_DIR "/data");
  const auto& object_infos = img_info.object_infos.value();

  for (std::size_t  i = 0; i < object_infos.size(); ++i) {
    const auto& obj_info = object_infos[i];
    fs::path shape_file = root_dir / fs::change_extension(obj_info.shape_file.value(), "ply");
    std::cout << "Loading " << shape_file << "\n";
    MeshData legacy_mesh = loadMeshFromPLY(shape_file.string());

    const Eigen::Vector3d half_dimension = obj_info.dimension.value() / 2;
    const Eigen::AlignedBox3d bbx(-half_dimension, half_dimension);
    Eigen::Affine3d affine_model_pose = pose_from_obj_info(obj_info, K_inv) * Eigen::UniformScaling<double>(bbx.diagonal().norm());
    renderer->modelDrawers().addItem(affine_model_pose.cast<float>(), legacy_mesh);
  }

  // Render
 viewer.render();

 {
   // Save color image
   QImage color_image = viewer.readColorBuffer();
   std::string image_file_path = fs::path(img_info.image_file.value()).filename().string();
   std::cout << image_file_path << "\n";
   color_image.save(QString::fromStdString(image_file_path));
   assert(color_image.width() == image_size[0]);
   assert(color_image.height() == image_size[1]);
 }



  return EXIT_SUCCESS;
}


