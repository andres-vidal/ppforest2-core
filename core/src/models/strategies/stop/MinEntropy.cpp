#include "models/strategies/stop/MinEntropy.hpp"

#include "models/strategies/NodeContext.hpp"
#include "stats/Entropy.hpp"
#include "utils/JsonReader.hpp"
#include "utils/UserError.hpp"

#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

using namespace ppforest2::types;

namespace ppforest2::stop {
  MinEntropy::MinEntropy(Feature threshold)
      : threshold(threshold) {
    user_error(threshold >= 0, "stop_min_entropy: threshold must be >= 0");
  }

  nlohmann::json MinEntropy::to_json() const {
    return {{"name", "min_entropy"}, {"threshold", threshold}};
  }

  std::string MinEntropy::display_name() const {
    std::ostringstream oss;
    oss << "Min entropy (" << std::defaultfloat << std::setprecision(6) << threshold << ")";
    return oss.str();
  }

  bool MinEntropy::compute(NodeContext const& ctx, stats::RNG& /*rng*/) const {
    std::vector<int> sizes;
    sizes.reserve(ctx.y.groups.size());

    for (auto const& g : ctx.y.groups) {
      sizes.push_back(ctx.y.group_size(g));
    }

    return stats::entropy(sizes) < static_cast<double>(threshold);
  }

  StopRule::Ptr min_entropy(Feature threshold) {
    return std::make_shared<MinEntropy>(threshold);
  }

  StopRule::Ptr MinEntropy::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "min_entropy"};
    r.only_keys({"name", "threshold"});
    return min_entropy(static_cast<Feature>(r.require_number("threshold", 0.0)));
  }
}
