#pragma once
#include <string>

namespace ps::quantization::fpx {

std::string compute_model_fingerprint(const std::string& model_dir);

}