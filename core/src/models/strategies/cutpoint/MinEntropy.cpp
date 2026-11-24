#include "models/strategies/cutpoint/MinEntropy.hpp"

#include "models/strategies/NodeContext.hpp"
#include "stats/Entropy.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::cutpoint {
  namespace {
    // Scores closer than this are equal: the lowest candidate wins, on
    // every platform, whatever the last bits of its logarithms.
    constexpr double SCORE_TOLERANCE = 1e-12;

    // A projected observation and the position of its group among the
    // node's groups.
    struct Observation {
      Feature value;
      std::size_t group;
    };

    std::vector<Observation> sorted_observations(NodeContext const& ctx) {
      auto const& y_part    = ctx.active_partition();
      auto const& projector = ctx.projector.value();

      std::vector<Observation> observations;
      observations.reserve(static_cast<std::size_t>(y_part.total_size()));

      std::size_t group = 0;
      for (auto const& g : y_part.groups) {
        FeatureVector const projected = ctx.x(y_part.group_indices(g), Eigen::indexing::all) * projector;

        for (Eigen::Index i = 0; i < projected.size(); ++i) {
          observations.push_back({projected(i), group});
        }

        ++group;
      }

      std::stable_sort(observations.begin(), observations.end(), [](Observation const& a, Observation const& b) {
        return a.value < b.value;
      });

      return observations;
    }

    // A value strictly above `lower` and at most `upper`, so that `lower`
    // is routed below the cutpoint and `upper` is not. The midpoint of two
    // adjacent floats rounds to one of them.
    Feature midpoint(Feature lower, Feature upper) {
      Feature const mid = lower + (upper - lower) / 2;
      return mid > lower ? mid : upper;
    }

    double weighted_entropy(std::vector<int> const& lower_sizes, std::vector<int> const& upper_sizes, int n_lower) {
      int const n_upper = static_cast<int>(std::accumulate(upper_sizes.begin(), upper_sizes.end(), 0));
      double const n    = n_lower + n_upper;

      return (n_lower * entropy(lower_sizes) + n_upper * entropy(upper_sizes)) / n;
    }

    std::optional<Feature> min_entropy_cutpoint(std::vector<Observation> const& observations, std::size_t n_groups) {
      std::vector<int> lower_sizes(n_groups, 0);
      std::vector<int> upper_sizes(n_groups, 0);

      for (auto const& observation : observations) {
        ++upper_sizes[observation.group];
      }

      std::optional<Feature> best;
      double best_score = std::numeric_limits<double>::infinity();

      for (std::size_t i = 0; i + 1 < observations.size(); ++i) {
        ++lower_sizes[observations[i].group];
        --upper_sizes[observations[i].group];

        Feature const value = observations[i].value;
        Feature const next  = observations[i + 1].value;

        if (!(value < next)) {
          continue;
        }

        double const score = weighted_entropy(lower_sizes, upper_sizes, static_cast<int>(i) + 1);

        if (score < best_score - SCORE_TOLERANCE) {
          best_score = score;
          best       = midpoint(value, next);
        }
      }

      return best;
    }
  }

  nlohmann::json MinEntropy::to_json() const {
    return {{"name", "min_entropy"}};
  }

  void MinEntropy::compute(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "MinEntropy requires projector on NodeContext");

    auto const observations = sorted_observations(ctx);
    auto const cutpoint     = min_entropy_cutpoint(observations, ctx.active_partition().groups.size());

    if (!cutpoint.has_value()) {
      ctx.aborted = true;
      return;
    }

    ctx.cutpoint = *cutpoint;
  }

  Cutpoint::Ptr min_entropy() {
    return std::make_shared<MinEntropy>();
  }

  Cutpoint::Ptr MinEntropy::from_json(nlohmann::json const& j) {
    JsonReader{j, "min_entropy"}.only_keys({"name"});
    return min_entropy();
  }
}
