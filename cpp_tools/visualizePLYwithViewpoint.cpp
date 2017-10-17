/**
 * @file visualizePLYwithViewpoint.cpp
 * @brief visualizePLYwithViewpoint
 *
 * @author Abhijit Kundu
 */

#include "CuteGL/Renderer/MultiObjectRenderer.h"
#include "CuteGL/Surface/WindowRenderViewer.h"
#include "CuteGL/Core/Config.h"
#include "CuteGL/IO/ImportPLY.h"
#include "CuteGL/Core/PoseUtils.h"

#include <Eigen/EulerAngles>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <QApplication>
#include <iostream>

int main(int argc, char **argv) {
  using namespace CuteGL;
  using namespace Eigen;
  namespace po = boost::program_options;
  namespace fs = boost::filesystem;

  float azimuth, elevation, tilt, distance;


  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("model,m",  po::value<std::string>()->default_value(CUTEGL_ASSETS_FOLDER "/car.ply"), "Path to ply model")
        ("azimuth,a",  po::value<float>(&azimuth)->default_value(0.0f), "Azimuth in Degrees)")
        ("elevation,e",  po::value<float>(&elevation)->default_value(0.0f), "Elevation in Degrees)")
        ("tilt,t",  po::value<float>(&tilt)->default_value(0.0f), "Tilt in Degrees)")
        ("distance,d",  po::value<float>(&distance)->default_value(3.0f), "Distance to object center)")
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

  // Get the input args in vector of strings
  const fs::path model_file(vm["model"].as<std::string>());
  std::cout << "Will load PLY file from " << model_file << "\n";


  QApplication app(argc, argv);

  std::unique_ptr<MultiObjectRenderer> renderer(new MultiObjectRenderer());

  WindowRenderViewer viewer(renderer.get());
  viewer.setBackgroundColor(0, 0, 0);

//  const int W = 960;
//  const int H = 540;
//  const float focal_length = 1050.0f;

  const int W = 1600;
  const int H = 800;
  const float focal_length = 1750.0f;

  viewer.resize(W, H);
  viewer.setSceneRadius(3.0f);

  viewer.camera().extrinsics() = getExtrinsicsFromViewPoint(0.0f, 0.0f, 0.0f, 2.0f);

  viewer.showAndWaitTillExposed();

  // Set camera intrinsics
  viewer.camera().intrinsics() = CuteGL::getGLPerspectiveProjection(focal_length, focal_length, 0.0f, W/2.0f, H/2.0f, W, H, 0.1f, 100.0f);

  renderer->modelDrawers().addItem(Affine3f::Identity(), loadMeshFromPLY(model_file.string()));
  renderer->modelDrawers().poses().front().setIdentity();

  const float pi_by_180 = M_PI / 180.0f;
  viewer.camera().extrinsics() = getExtrinsicsFromViewPoint(azimuth * pi_by_180,
                                                            elevation * pi_by_180,
                                                            tilt * pi_by_180,
                                                            distance);

  app.exec();
}


