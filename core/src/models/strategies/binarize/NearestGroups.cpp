#include "models/strategies/binarize/NearestGroups.hpp"

#include "models/strategies/binarize/GroupMeans.hpp"
#include "models/strategies/NodeContext.hpp"
#include "stats/Entropy.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <limits>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::binarize {
  namespace {
    using Gap = NearestGroups::Gap;

    // Sums closer than this are equal: the lowest gap wins, on every
    // platform, whatever the last bits of its logarithms.
    constexpr double ENTROPY_TOLERANCE = 1e-12;

    char const* name_of(Gap gap) {
      return gap == Gap::Largest ? "largest" : "min_entropy";
    }

    // Sum of the entropies of the observations at or below `midpoint` and
    // of those above it, each counted by group.
    double entropy_around(NodeContext const& ctx, FeatureVector const& projected, Feature midpoint) {
      std::vector<int> lower_sizes;
      std::vector<int> upper_sizes;

      for (GroupId const group : ctx.y.groups) {
        int const size  = ctx.y.group_size(group);
        int const lower = static_cast<int>((ctx.y.group(projected, group).array() <= midpoint).count());

        lower_sizes.push_back(lower);
        upper_sizes.push_back(size - lower);
      }

      return entropy(lower_sizes) + entropy(upper_sizes);
    }

    // Position in `means` of the first group above the gap whose midpoint
    // has the lowest entropy around it; the lowest such gap among equals.
    std::size_t
    min_entropy_edge(NodeContext const& ctx, FeatureVector const& projected, std::vector<GroupMean> const& means) {
      double lowest    = std::numeric_limits<double>::infinity();
      std::size_t edge = 0;

      for (std::size_t i = 0; i + 1 < means.size(); i++) {
        Feature const midpoint = (means[i].mean + means[i + 1].mean) / 2;
        double const around    = entropy_around(ctx, projected, midpoint);

        if (around < lowest - ENTROPY_TOLERANCE) {
          lowest = around;
          edge   = i + 1;
        }
      }

      return edge;
    }
  }

  NearestGroups::NearestGroups(Gap gap)
      : gap(gap) {}

  nlohmann::json NearestGroups::to_json() const {
    return {{"name", "nearest_groups"}, {"gap", name_of(gap)}};
  }

  std::string NearestGroups::display_name() const {
    return std::string("Nearest groups (gap: ") + name_of(gap) + ")";
  }

  void NearestGroups::compute(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "NearestGroups requires projector on NodeContext");
    FeatureVector const projected = ctx.x * ctx.projector.value();

    auto const means = sorted_group_means(ctx, projected);
    auto const edge  = gap == Gap::Largest ? largest_gap_edge(means) : min_entropy_edge(ctx, projected, means);

    ctx.y_bin.emplace(binarize_at(ctx.y, means, edge));

    // No gap: every group is on one side, and the base class aborts the node.
    if (edge == 0) {
      return;
    }

    ctx.y_projection.emplace(ctx.y.subset({means[edge - 1].group, means[edge].group}));
  }

  Binarization::Ptr nearest_groups(Gap gap) {
    return std::make_shared<NearestGroups>(gap);
  }

  Binarization::Ptr NearestGroups::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "nearest_groups"};
    r.only_keys({"name", "gap"});

    // The gap is optional: the largest, as in largest_gap, by default.
    bool const min_entropy = r.contains("gap") && r.require_enum("gap", {"largest", "min_entropy"}) == "min_entropy";

    return nearest_groups(min_entropy ? Gap::MinEntropy : Gap::Largest);
  }
}
