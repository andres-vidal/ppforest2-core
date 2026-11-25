#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the entropy of the node's groups is below a threshold.
   *
   * The entropy is `-sum(p_g * log(p_g))` over the shares of the groups
   * in the node, with the natural logarithm: 0 for a pure node and
   * `log(G)` for G groups of equal size. A node that is almost pure is
   * left as a leaf instead of being split to isolate a few observations.
   */
  class MinEntropy : public StopRule {
  public:
    types::Feature threshold;

    explicit MinEntropy(types::Feature threshold);

    static StopRule::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    PPFOREST2_REGISTER_STRATEGY(StopRule, "min_entropy")
    PPFOREST2_REGISTER_PRIMARY_PARAM("min_entropy", "threshold")

  protected:
    bool compute(NodeContext const& ctx, stats::RNG& rng) const override;
  };
}
