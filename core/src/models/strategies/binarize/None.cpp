#include "models/strategies/binarize/None.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::binarize {
  nlohmann::json None::to_json() const {
    return {{"name", "none"}};
  }

  void None::compute(NodeContext& /*ctx*/, stats::RNG& /*rng*/) const {}

  Binarization::Ptr none() {
    return std::make_shared<None>();
  }

  Binarization::Ptr None::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "none"};
    r.only_keys({"name"});
    return none();
  }
}
