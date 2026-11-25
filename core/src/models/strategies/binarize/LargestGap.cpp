#include "models/strategies/binarize/LargestGap.hpp"

#include "models/strategies/binarize/GroupMeans.hpp"
#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::binarize {
  nlohmann::json LargestGap::to_json() const {
    return {{"name", "largest_gap"}};
  }

  void LargestGap::compute(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "LargestGap requires projector on NodeContext");
    FeatureVector const projected = ctx.x * ctx.projector.value();

    auto const means = sorted_group_means(ctx, projected);

    ctx.y_bin.emplace(binarize_at(ctx.y, means, largest_gap_edge(means)));
  }

  Binarization::Ptr largest_gap() {
    return std::make_shared<LargestGap>();
  }

  Binarization::Ptr LargestGap::from_json(nlohmann::json const& j) {
    JsonReader{j, "largest_gap"}.only_keys({"name"});
    return largest_gap();
  }
}
