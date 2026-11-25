#pragma once

#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::binarize {
  /**
   * @brief No binarization: the node keeps all its groups.
   *
   * The projection is optimized once, over every group in the node, and
   * the cutpoint and the grouping work on that multiclass projection. It
   * is the binarization of trees that may split a group more than once
   * (`cutpoint::MinEntropy` with `grouping::ByObservation`).
   *
   * Unlike `Disabled`, which fails when a node has more than two groups,
   * this strategy is meant to run on such nodes and leave them as they are.
   */
  class None : public Binarization {
  public:
    static Binarization::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "None"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }
    bool binarizes() const override { return false; }

    PPFOREST2_REGISTER_STRATEGY(Binarization, "none")

  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };
}
