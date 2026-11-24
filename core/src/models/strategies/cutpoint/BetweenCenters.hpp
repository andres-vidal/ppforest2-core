#pragma once

#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/Types.hpp"

#include <string>

namespace ppforest2::cutpoint {
  /**
   * @brief Split cutpoint between the centers of the two projected groups.
   *
   * With centers `c1`, `c2` and weights `w1`, `w2` of the two groups in
   * projected space, the cutpoint is `(c1 * w2 + c2 * w1) / (w1 + w2)`:
   * each center is weighted by the other group's weight, so the cutpoint
   * moves toward the group with the smaller weight.
   *
   * `center` picks the group center and `weight` what it is weighted by.
   * Together they cover the eight split rules of PPforest (its `rule`
   * argument):
   *
   * | center | weight         | PPforest rule |
   * |--------|----------------|---------------|
   * | mean   | none           | 1 (same as MeanOfMeans) |
   * | mean   | size           | 2 |
   * | mean   | spread         | 3 (standard deviation) |
   * | mean   | standard_error | 4 (standard deviation / sqrt(n)) |
   * | median | none           | 5 |
   * | median | size           | 6 |
   * | median | spread         | 7 (interquartile range) |
   * | median | standard_error | 8 (interquartile range / sqrt(n)) |
   *
   * The spread of a group is its standard deviation when the center is
   * the mean and its interquartile range when it is the median. When
   * either group has no spread, `spread` falls back to `none` and
   * `standard_error` to `size`, as PPforest does.
   *
   * In JSON and on the command line, `center` defaults to `mean` and
   * `weight` to `none`.
   */
  class BetweenCenters : public Cutpoint {
  public:
    enum class Center { Mean, Median };
    enum class Weight { None, Size, Spread, StandardError };

    Center center;
    Weight weight;

    BetweenCenters(Center center, Weight weight);

    static Cutpoint::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    PPFOREST2_REGISTER_STRATEGY(Cutpoint, "between_centers")

  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };

  /** @brief Factory function for a cutpoint between two group centers. */
  Cutpoint::Ptr between_centers(BetweenCenters::Center center, BetweenCenters::Weight weight);
}
