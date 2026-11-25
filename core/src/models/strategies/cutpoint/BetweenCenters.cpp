#include "models/strategies/cutpoint/BetweenCenters.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::cutpoint {
  namespace {
    using Center = BetweenCenters::Center;
    using Weight = BetweenCenters::Weight;

    char const* name_of(Center center) {
      return center == Center::Mean ? "mean" : "median";
    }

    char const* name_of(Weight weight) {
      switch (weight) {
        case Weight::None: return "none";
        case Weight::Size: return "size";
        case Weight::Spread: return "spread";
        case Weight::StandardError: return "standard_error";
      }

      return "none";
    }

    Center center_from(std::string const& name) {
      return name == "mean" ? Center::Mean : Center::Median;
    }

    Weight weight_from(std::string const& name) {
      if (name == "size") {
        return Weight::Size;
      }

      if (name == "spread") {
        return Weight::Spread;
      }

      return name == "standard_error" ? Weight::StandardError : Weight::None;
    }

    // Quantile of sorted values by linear interpolation between order
    // statistics (R's default, type 7), so the median and the
    // interquartile range match the ones PPforest takes from R.
    Feature quantile(std::vector<Feature> const& sorted, Feature p) {
      Feature const h      = static_cast<Feature>(sorted.size() - 1) * p;
      auto const lower     = static_cast<std::size_t>(std::floor(h));
      auto const upper     = std::min(lower + 1, sorted.size() - 1);
      Feature const within = h - static_cast<Feature>(lower);

      return sorted[lower] + within * (sorted[upper] - sorted[lower]);
    }

    // Sample standard deviation; 0 for a single value.
    Feature standard_deviation(std::vector<Feature> const& values, Feature mean) {
      if (values.size() < 2) {
        return 0;
      }

      Feature sum_of_squares = 0;
      for (Feature const v : values) {
        sum_of_squares += (v - mean) * (v - mean);
      }

      return std::sqrt(sum_of_squares / static_cast<Feature>(values.size() - 1));
    }

    // What the cutpoint needs to know about one projected group.
    struct GroupSummary {
      Feature center = 0;
      Feature spread = 0;
      Feature size   = 0;
    };

    GroupSummary summarize(FeatureVector const& projected, Center center) {
      std::vector<Feature> values(projected.data(), projected.data() + projected.size());
      std::sort(values.begin(), values.end());

      GroupSummary summary;
      summary.size = static_cast<Feature>(values.size());

      // A constant group is centered on its value with no spread. Its float
      // mean can differ from that value by rounding, which would give it a
      // small spread and keep the rule without spread from applying.
      if (values.front() == values.back()) {
        summary.center = values.front();
        return summary;
      }

      if (center == Center::Mean) {
        summary.center = projected.mean();
        summary.spread = standard_deviation(values, summary.center);
      } else {
        summary.center = quantile(values, 0.5F);
        summary.spread = quantile(values, 0.75F) - quantile(values, 0.25F);
      }

      return summary;
    }

    Feature weight_of(GroupSummary const& group, Weight weight) {
      switch (weight) {
        case Weight::None: return 1;
        case Weight::Size: return group.size;
        case Weight::Spread: return group.spread;
        case Weight::StandardError: return group.spread / std::sqrt(group.size);
      }

      return 1;
    }

    // A group without spread has weight 0, which would put the cutpoint on
    // the other group's center; the rule without the spread is used instead.
    Weight without_spread(Weight weight) {
      return weight == Weight::StandardError ? Weight::Size : Weight::None;
    }

    bool uses_spread(Weight weight) {
      return weight == Weight::Spread || weight == Weight::StandardError;
    }
  }

  BetweenCenters::BetweenCenters(Center center, Weight weight)
      : center(center)
      , weight(weight) {}

  nlohmann::json BetweenCenters::to_json() const {
    return {{"name", "between_centers"}, {"center", name_of(center)}, {"weight", name_of(weight)}};
  }

  std::string BetweenCenters::display_name() const {
    return std::string("Between ") + name_of(center) + "s (weight: " + name_of(weight) + ")";
  }

  void BetweenCenters::compute(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "BetweenCenters requires projector on NodeContext");
    auto const& y_part    = ctx.active_partition();
    auto const& projector = ctx.projector.value();
    auto const g1         = *y_part.groups.begin();
    auto const g2         = *std::next(y_part.groups.begin());

    FeatureVector const projected_1 = ctx.x(y_part.group_indices(g1), Eigen::indexing::all) * projector;
    FeatureVector const projected_2 = ctx.x(y_part.group_indices(g2), Eigen::indexing::all) * projector;

    GroupSummary const s1 = summarize(projected_1, center);
    GroupSummary const s2 = summarize(projected_2, center);

    bool const no_spread = uses_spread(weight) && (s1.spread == 0 || s2.spread == 0);
    Weight const applied = no_spread ? without_spread(weight) : weight;

    Feature const w1 = weight_of(s1, applied);
    Feature const w2 = weight_of(s2, applied);

    ctx.cutpoint = (s1.center * w2 + s2.center * w1) / (w1 + w2);
  }

  Cutpoint::Ptr between_centers(Center center, Weight weight) {
    return std::make_shared<BetweenCenters>(center, weight);
  }

  Cutpoint::Ptr BetweenCenters::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "between_centers"};
    r.only_keys({"name", "center", "weight"});

    // Both are optional: the mean of means is the cutpoint by default.
    auto const center = r.contains("center") ? center_from(r.require_enum("center", {"mean", "median"})) : Center::Mean;
    auto const weight = r.contains("weight")
                            ? weight_from(r.require_enum("weight", {"none", "size", "spread", "standard_error"}))
                            : Weight::None;

    return between_centers(center, weight);
  }
}
