#pragma once

#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/Types.hpp"

#include <string>

namespace ppforest2::binarize {
  /**
   * @brief Binarization whose splitting projection uses the two groups
   *        nearest the gap.
   *
   * The groups are ordered by projected mean and divided at a gap between
   * two consecutive means, as in `LargestGap`. The projection that splits
   * the node is then optimized on the two groups next to that gap, one
   * from each side, instead of on the two sides as a whole. When a side
   * gathers groups that are far apart, the projection that separates the
   * sides as a whole is pulled toward the distant groups; the two groups
   * that meet at the boundary give a projection that follows it. This is
   * the first PPtree extension of da Silva, Cook and Lee (2026), as
   * implemented in PPtreeExt (`PPtreeExt_split`).
   *
   * The cutpoint and the grouping still work on the two sides: the
   * cutpoint is located from all their observations, projected with the
   * new projection, and every group follows its side.
   *
   * `gap` chooses the gap:
   * - `Largest`: the largest gap between consecutive means.
   * - `MinEntropy`: the gap whose midpoint leaves the two sides with the
   *   lowest sum of entropies, counting the observations (not the groups)
   *   on each side of the midpoint.
   */
  class NearestGroups : public Binarization {
  public:
    enum class Gap { Largest, MinEntropy };

    Gap gap;

    explicit NearestGroups(Gap gap);

    static Binarization::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    PPFOREST2_REGISTER_STRATEGY(Binarization, "nearest_groups")

  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };

  /** @brief Factory function for the nearest-groups binarization. */
  Binarization::Ptr nearest_groups(NearestGroups::Gap gap = NearestGroups::Gap::Largest);
}
