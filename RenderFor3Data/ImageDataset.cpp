/**
 * @file ImageDataset.cpp
 * @brief ImageDataset
 *
 * @author Abhijit Kundu
 */

#include "ImageDataset.h"
#include <fstream>
#include <boost/optional/optional_io.hpp>

namespace nlohmann {

/// Serialization support for boost::optional
template<typename T>
struct adl_serializer<boost::optional<T>> {
  static void to_json(json& j, const boost::optional<T>& opt) {
    if (opt == boost::none) {
      j = nullptr;
    } else {
      j = *opt;  // this will call adl_serializer<T>::to_json which will
                 // find the free function to_json in T's namespace!
    }
  }

  static void from_json(const json& j, boost::optional<T>& opt) {
    if (j.is_null()) {
      opt = boost::none;
    } else {
      opt = j.get<T>();  // same as above, but with
                         // adl_serializer<T>::from_json
    }
  }
};

} // end namespace nlohmann

namespace Eigen {

template<typename Derived>
void to_json(nlohmann::json& json, const Eigen::MatrixBase<Derived>& mat) {

  if (mat.rows() == 1 || mat.cols() == 1) {
    for (Index i = 0; i < mat.rows(); ++i) {
      for (Index j = 0; j < mat.cols(); ++j) {
        json.push_back(mat(i, j));
      }
    }
  } else {
    for (Index i = 0; i < mat.rows(); ++i) {
      nlohmann::json row = nlohmann::json::array();
      for (Index j = 0; j < mat.cols(); ++j) {
        row.push_back(mat(i, j));
      }
      json.push_back(row);
    }
  }
}

template<typename Derived>
void from_json(const nlohmann::json& json, MatrixBase<Derived> const &mat_) {
  using Scalar = typename Derived::Scalar;
  MatrixBase<Derived>& mat = const_cast< MatrixBase<Derived>& >(mat_);

  if (Derived::IsVectorAtCompileTime) {
    using Vec = std::vector<Scalar>;
    Vec vec = json.get<Vec>();
    using PlainObject = typename Derived::PlainObject;

    Index rows = Derived::MaxRowsAtCompileTime == Dynamic ? vec.size() : 1;
    Index cols = Derived::MaxColsAtCompileTime == Dynamic ? vec.size() : 1;
    mat = Map<PlainObject>(vec.data(), rows, cols);
  }
  else {
    using VecOfVec = std::vector<std::vector<Scalar>>;
    VecOfVec vec_of_vec = json.get<VecOfVec>();

    const Index rows = vec_of_vec.size();
    const Index cols = vec_of_vec.at(0).size();

    mat.resize(rows, cols);
    for (Index i = 0; i < rows; ++i) {
      for (Index j = 0; j < cols; ++j) {
        mat(i, j) = vec_of_vec[i][j];
      }
    }
  }
}

}  // namespace Eigen

template<typename KeyType, typename T, std::size_t N>
void from_json_if_present(const nlohmann::json& j, const KeyType& key, boost::optional<std::array<T, N>>& opt) {
  auto it = j.find(key);
  if (it != j.end()) {
    using VectorT = std::vector<T>;
    VectorT vec = it->template get<VectorT>();
    if (vec.size() == N) {
      opt.emplace();
      std::copy_n(vec.begin(), N, opt.value().begin());
      return;
    }
  }
  opt = boost::none;
}

template<typename KeyType, typename T>
void from_json_if_present(const nlohmann::json& j, const KeyType& key, boost::optional<T>& opt) {
  auto it = j.find(key);
  if (it != j.end()) {
    opt = it->template get<T>();
    return;
  }
  opt = boost::none;
}

template<typename KeyType, typename T>
void add_to_json_if_present(nlohmann::json& j, const KeyType& key, const boost::optional<T>& opt) {
  if (opt)
    j[key] = opt.value();
}

ImageDataset loadImageDatasetFromJson(const std::string& filepath) {
  nlohmann::json dataset_json;
  {
    std::ifstream file(filepath.c_str());
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open File from " + filepath);
    }
    file >> dataset_json;
  }
  return dataset_json;
}

void saveImageDatasetToJson(const ImageDataset& image_dataset, const std::string& filepath) {
  {
    std::ofstream file(filepath.c_str());
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open File from " + filepath);
    }
    file << nlohmann::json(image_dataset).dump(2) << "\n";
    file.close();
  }
}

void to_json(nlohmann::json& j, const ImageDataset& dataset) {
  j = nlohmann::json {
    { "name", dataset.name },
    { "rootdir", dataset.rootdir.string() },
    { "image_infos", dataset.image_infos }
  };
}

void from_json(const nlohmann::json& j, ImageDataset& dataset) {
  dataset.name = j["name"].get<std::string>();
  dataset.rootdir = j["rootdir"].get<std::string>();
  dataset.image_infos = j["image_infos"].get<std::vector<ImageInfo>>();
}


void to_json(nlohmann::json& j, const ImageInfo& p) {
  j = nlohmann::json {
    { "image_file", p.image_file },
    { "segm_file", p.segm_file },
    { "image_size", p.image_size },
    { "image_intrinsic", p.image_intrinsic },
    { "object_infos", p.object_infos }
  };
}

void from_json(const nlohmann::json& j, ImageInfo& p) {
  from_json_if_present(j, "image_file", p.image_file);
  from_json_if_present(j, "segm_file", p.segm_file);
  from_json_if_present(j, "image_size", p.image_size);
  from_json_if_present(j, "image_intrinsic", p.image_intrinsic);
  from_json_if_present(j, "object_infos", p.object_infos);
}

void to_json(nlohmann::json& j, const ImageObjectInfo& p) {
  add_to_json_if_present(j, "id", p.id);
  add_to_json_if_present(j, "category", p.category);
  add_to_json_if_present(j, "truncation", p.truncation);
  add_to_json_if_present(j, "occlusion", p.occlusion);
  add_to_json_if_present(j, "dimension", p.dimension);
  add_to_json_if_present(j, "bbx_visible", p.bbx_visible);
  add_to_json_if_present(j, "bbx_amodal", p.bbx_amodal);
  add_to_json_if_present(j, "viewpoint", p.viewpoint);
  add_to_json_if_present(j, "center_proj", p.center_proj);
  add_to_json_if_present(j, "center_dist", p.center_dist);
  add_to_json_if_present(j, "pose_param", p.pose_param);
  add_to_json_if_present(j, "shape_param", p.shape_param);
  add_to_json_if_present(j, "shape_file", p.shape_file);
}

void from_json(const nlohmann::json& j, ImageObjectInfo& p) {
  from_json_if_present(j, "id", p.id);
  from_json_if_present(j, "category", p.category);
  from_json_if_present(j, "truncation", p.truncation);
  from_json_if_present(j, "occlusion", p.occlusion);
  from_json_if_present(j, "dimension", p.dimension);
  from_json_if_present(j, "bbx_visible", p.bbx_visible);
  from_json_if_present(j, "bbx_amodal", p.bbx_amodal);
  from_json_if_present(j, "viewpoint", p.viewpoint);
  from_json_if_present(j, "center_proj", p.center_proj);
  from_json_if_present(j, "center_dist", p.center_dist);
  from_json_if_present(j, "pose_param", p.pose_param);
  from_json_if_present(j, "shape_param", p.shape_param);
  from_json_if_present(j, "shape_file", p.shape_file);
}


template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& val) {
  os << "[";
  for (const auto& s : val)
    os << s << ", ";
  os << "\b\b]";
  return os;
}

template<typename T, typename A>
std::ostream& operator<<(std::ostream& os, const std::vector<T, A>& val) {
  os << "[";
  for (const auto& s : val)
    os << s << ", ";
  os << "\b\b]";
  return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const boost::optional<T>& opt) {
  if (opt == boost::none) {
    os << "none";
  } else
    os << opt.value();
  return os;
}


