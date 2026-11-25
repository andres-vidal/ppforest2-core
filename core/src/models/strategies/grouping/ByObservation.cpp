#include "models/strategies/grouping/ByObservation.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"
#include "utils/RangeVector.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::grouping {
  namespace {
    /**
     * @brief Move the rows of [start, start + n) below the cutpoint to the front.
     *
     * Reorders `x` and `y` in place, keeping the relative order on each
     * side, and returns how many rows are below the cutpoint.
     */
    int lower_rows_first(
        FeatureMatrix& x, OutcomeVector& y, pp::Projector const& projector, Feature cutpoint, int start, int n
    ) {
      FeatureVector const projected = x.middleRows(start, n) * projector;

      std::vector<int> order = utils::range_vector(n);
      auto const first_upper =
          std::stable_partition(order.begin(), order.end(), [&](int i) { return projected(i) < cutpoint; });

      x.middleRows(start, n) = x.middleRows(start, n)(order, Eigen::indexing::all).eval();
      y.segment(start, n)    = y.segment(start, n)(order).eval();

      return static_cast<int>(std::distance(order.begin(), first_upper));
    }
  }

  nlohmann::json ByObservation::to_json() const {
    return {{"name", "by_observation"}};
  }

  GroupPartition ByObservation::compute_init(OutcomeVector const& y) const {
    return GroupPartition(y);
  }

  void ByObservation::compute(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "ByObservation requires projector on NodeContext");
    invariant(ctx.cutpoint.has_value(), "ByObservation requires cutpoint on NodeContext");

    GroupPartition::SplitSizes lower_sizes;
    int n_lower = 0;

    for (auto const& g : ctx.y.groups) {
      int const in_lower = lower_rows_first(
          ctx.x, ctx.y_vec, ctx.projector.value(), ctx.cutpoint.value(), ctx.y.group_start(g), ctx.y.group_size(g)
      );

      lower_sizes[g] = in_lower;
      n_lower += in_lower;
    }

    // Every observation on one side: both children get the whole node, and
    // the no-progress check in Grouping::split aborts it.
    if (n_lower == 0 || n_lower == ctx.y.total_size()) {
      ctx.lower_y_part.emplace(ctx.y);
      ctx.upper_y_part.emplace(ctx.y);
      return;
    }

    auto children = ctx.y.split(lower_sizes);
    ctx.lower_y_part.emplace(std::move(children.first));
    ctx.upper_y_part.emplace(std::move(children.second));
  }

  Grouping::Ptr by_observation() {
    return std::make_shared<ByObservation>();
  }

  Grouping::Ptr ByObservation::from_json(nlohmann::json const& j) {
    JsonReader{j, "by_observation"}.only_keys({"name"});
    return by_observation();
  }
}
