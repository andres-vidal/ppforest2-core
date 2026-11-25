#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the node holds at most a share of the training data.
   *
   * The node is a leaf when its size divided by the number of training
   * observations is at most `proportion`. Unlike `MinSize`, the limit
   * follows the size of the data.
   */
  class MinProportion : public StopRule {
  public:
    types::Feature proportion;

    explicit MinProportion(types::Feature proportion);

    static StopRule::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    PPFOREST2_REGISTER_STRATEGY(StopRule, "min_proportion")
    PPFOREST2_REGISTER_PRIMARY_PARAM("min_proportion", "proportion")

  protected:
    bool compute(NodeContext const& ctx, stats::RNG& rng) const override;
  };
}
