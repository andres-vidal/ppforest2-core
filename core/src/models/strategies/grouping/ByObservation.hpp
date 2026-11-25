#pragma once

#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::grouping {
  /**
   * @brief Grouping that routes each observation by its own projection.
   *
   * `ByLabel` sends every observation of a group to the same child, so a
   * tree has one leaf per group. This strategy sends each observation to
   * the lower child when its projection is below the cutpoint and to the
   * upper child otherwise, whatever its group. A group can therefore be
   * divided between the two children and end up in several leaves, which
   * lets a tree follow boundaries that a single split per group cannot
   * (da Silva, Cook and Lee, 2026).
   *
   * The observations keep their labels: each child's partition holds the
   * groups that reached it. Rows are reordered in place inside each
   * group's block (lower rows first), so the context's `x` and `y_vec`
   * must be mutable.
   *
   * A tree grown this way is not bounded by the number of groups; it
   * needs a stop rule besides `PureNode` to stay small.
   */
  class ByObservation : public Grouping {
  public:
    static Grouping::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "By observation"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }
    bool requires_two_groups() const override { return false; }

    PPFOREST2_REGISTER_STRATEGY(Grouping, "by_observation")

  protected:
    /** @brief One group per label, as `ByLabel`. */
    stats::GroupPartition compute_init(types::OutcomeVector const& y) const override;

    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };
}
