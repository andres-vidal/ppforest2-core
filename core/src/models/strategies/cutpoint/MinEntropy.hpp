#pragma once

#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/Types.hpp"

namespace ppforest2::cutpoint {
  /**
   * @brief Split cutpoint that minimizes the entropy of the two sides.
   *
   * Every midpoint between two consecutive distinct projected values is a
   * candidate. For each one, the observations below it form the lower side
   * and the rest the upper side, and the candidate's score is
   *
   *     (n_lower * E(lower) + n_upper * E(upper)) / n
   *
   * with `E` the entropy of a side's group shares. The cutpoint is the
   * candidate with the lowest score; among equals, the lowest candidate.
   *
   * The groups are those of the active partition, so the strategy works
   * on a node with any number of groups: it does not need a binarization,
   * and it is the cutpoint of trees that may split a group more than once
   * (da Silva, Cook and Lee, 2026).
   *
   * The node is aborted when all its projected values are equal, since no
   * cutpoint separates them.
   */
  class MinEntropy : public Cutpoint {
  public:
    static Cutpoint::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Min entropy"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }
    bool requires_two_groups() const override { return false; }

    PPFOREST2_REGISTER_STRATEGY(Cutpoint, "min_entropy")

  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };
}
