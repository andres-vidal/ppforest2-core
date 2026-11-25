#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/cutpoint/BetweenCenters.hpp"
#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/cutpoint/MeanOfMeans.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <cmath>

using namespace ppforest2;
using namespace ppforest2::cutpoint;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

using Center = BetweenCenters::Center;
using Weight = BetweenCenters::Weight;

namespace {
  // One feature, projected onto itself, so the projected values are the data.
  //
  // Group 0: 0, 1, 2, 9      n = 4, mean = 3, median = 1.5, sd = sqrt(50 / 3), IQR = 3
  // Group 1: 10, 12, 20      n = 3, mean = 14, median = 12, sd = sqrt(28), IQR = 5
  //
  // The quartiles are R's default (type 7): quantile(c(0, 1, 2, 9), c(.25, .75))
  // is 0.75 and 3.75; quantile(c(10, 12, 20), c(.25, .75)) is 11 and 16.
  Feature cutpoint_of(Center center, Weight weight) {
    NodeContextFixture f(MAT(Feature, rows(7), 0, 1, 2, 9, 10, 12, 20), VEC(GroupId, 0, 0, 0, 0, 1, 1, 1));
    f.ctx.projector = VEC(Feature, 1);

    BetweenCenters(center, weight).locate(f.ctx, f.rng);

    return f.ctx.cutpoint.value();
  }

  // Both groups are constant, so neither has spread.
  Feature cutpoint_without_spread(Center center, Weight weight) {
    NodeContextFixture f(MAT(Feature, rows(5), 2, 2, 2, 2, 8), VEC(GroupId, 0, 0, 0, 0, 1));
    f.ctx.projector = VEC(Feature, 1);

    BetweenCenters(center, weight).locate(f.ctx, f.rng);

    return f.ctx.cutpoint.value();
  }

  // Group 0 is seven values of 0.1, whose float mean is not exactly 0.1.
  // Group 1 is 10, 12.
  Feature cutpoint_with_rounded_mean(Weight weight) {
    NodeContextFixture f(
        MAT(Feature, rows(9), 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 10, 12), VEC(GroupId, 0, 0, 0, 0, 0, 0, 0, 1, 1)
    );
    f.ctx.projector = VEC(Feature, 1);

    BetweenCenters(Center::Mean, weight).locate(f.ctx, f.rng);

    return f.ctx.cutpoint.value();
  }
}

TEST(CutpointBetweenCenters, FromJsonRoundTrip) {
  json const j = {{"name", "between_centers"}, {"center", "median"}, {"weight", "standard_error"}};

  EXPECT_EQ(BetweenCenters::from_json(j)->to_json(), j);
}

TEST(CutpointBetweenCenters, FromJsonDefaultsToMeanWithoutWeight) {
  json const j        = {{"name", "between_centers"}};
  json const expected = {{"name", "between_centers"}, {"center", "mean"}, {"weight", "none"}};

  EXPECT_EQ(BetweenCenters::from_json(j)->to_json(), expected);
}

TEST(CutpointBetweenCenters, FromJsonUnknownCenterThrows) {
  json const j = {{"name", "between_centers"}, {"center", "mode"}};
  EXPECT_THROW(BetweenCenters::from_json(j), std::runtime_error);
}

TEST(CutpointBetweenCenters, FromJsonUnknownWeightThrows) {
  json const j = {{"name", "between_centers"}, {"weight", "variance"}};
  EXPECT_THROW(BetweenCenters::from_json(j), std::runtime_error);
}

TEST(CutpointBetweenCenters, FromJsonUnknownParamThrows) {
  json const j = {{"name", "between_centers"}, {"extra", 1}};
  EXPECT_THROW(BetweenCenters::from_json(j), std::runtime_error);
}

TEST(CutpointBetweenCenters, RegistryLookup) {
  json const j = {{"name", "between_centers"}, {"center", "mean"}, {"weight", "size"}};

  auto strategy = Cutpoint::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(CutpointBetweenCenters, DisplayNameNamesCenterAndWeight) {
  EXPECT_EQ(BetweenCenters(Center::Median, Weight::Size).display_name(), "Between medians (weight: size)");
}

/* PPforest rule 1: (3 + 14) / 2. */
TEST(CutpointBetweenCenters, MeanWithoutWeightIsMeanOfMeans) {
  EXPECT_FLOAT_EQ(cutpoint_of(Center::Mean, Weight::None), 8.5F);

  NodeContextFixture f(MAT(Feature, rows(7), 0, 1, 2, 9, 10, 12, 20), VEC(GroupId, 0, 0, 0, 0, 1, 1, 1));
  f.ctx.projector = VEC(Feature, 1);
  MeanOfMeans().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(cutpoint_of(Center::Mean, Weight::None), f.ctx.cutpoint.value());
}

/* PPforest rule 2: each mean weighted by the other group's size, (3 * 3 + 14 * 4) / 7. */
TEST(CutpointBetweenCenters, MeanWeightedBySize) {
  EXPECT_FLOAT_EQ(cutpoint_of(Center::Mean, Weight::Size), 65.0F / 7.0F);
}

/* PPforest rule 3: (3 * sd1 + 14 * sd0) / (sd0 + sd1). */
TEST(CutpointBetweenCenters, MeanWeightedBySpread) {
  Feature const sd0 = std::sqrt(50.0F / 3.0F);
  Feature const sd1 = std::sqrt(28.0F);

  EXPECT_FLOAT_EQ(cutpoint_of(Center::Mean, Weight::Spread), (3 * sd1 + 14 * sd0) / (sd0 + sd1));
}

/* PPforest rule 4: as rule 3 with sd / sqrt(n). */
TEST(CutpointBetweenCenters, MeanWeightedByStandardError) {
  Feature const se0 = std::sqrt(50.0F / 3.0F) / std::sqrt(4.0F);
  Feature const se1 = std::sqrt(28.0F) / std::sqrt(3.0F);

  EXPECT_FLOAT_EQ(cutpoint_of(Center::Mean, Weight::StandardError), (3 * se1 + 14 * se0) / (se0 + se1));
}

/* PPforest rule 5: (1.5 + 12) / 2. */
TEST(CutpointBetweenCenters, MedianWithoutWeight) {
  EXPECT_FLOAT_EQ(cutpoint_of(Center::Median, Weight::None), 6.75F);
}

/* PPforest rule 6: (1.5 * 3 + 12 * 4) / 7. */
TEST(CutpointBetweenCenters, MedianWeightedBySize) {
  EXPECT_FLOAT_EQ(cutpoint_of(Center::Median, Weight::Size), 52.5F / 7.0F);
}

/* PPforest rule 7: (1.5 * IQR1 + 12 * IQR0) / (IQR0 + IQR1) = (1.5 * 5 + 12 * 3) / 8. */
TEST(CutpointBetweenCenters, MedianWeightedBySpread) {
  EXPECT_FLOAT_EQ(cutpoint_of(Center::Median, Weight::Spread), 43.5F / 8.0F);
}

/* PPforest rule 8: as rule 7 with IQR / sqrt(n). */
TEST(CutpointBetweenCenters, MedianWeightedByStandardError) {
  Feature const w0 = 3.0F / std::sqrt(4.0F);
  Feature const w1 = 5.0F / std::sqrt(3.0F);

  EXPECT_FLOAT_EQ(cutpoint_of(Center::Median, Weight::StandardError), (1.5F * w1 + 12 * w0) / (w0 + w1));
}

/* Without spread in a group, `spread` is the unweighted rule: (2 + 8) / 2. */
TEST(CutpointBetweenCenters, SpreadFallsBackToNoWeight) {
  EXPECT_FLOAT_EQ(cutpoint_without_spread(Center::Mean, Weight::Spread), 5.0F);
  EXPECT_FLOAT_EQ(cutpoint_without_spread(Center::Median, Weight::Spread), 5.0F);
}

/* Without spread in a group, `standard_error` is the size rule: (2 * 1 + 8 * 4) / 5. */
TEST(CutpointBetweenCenters, StandardErrorFallsBackToSize) {
  EXPECT_FLOAT_EQ(cutpoint_without_spread(Center::Mean, Weight::StandardError), 6.8F);
  EXPECT_FLOAT_EQ(cutpoint_without_spread(Center::Median, Weight::StandardError), 6.8F);
}

/*
 * Seven values of 0.1 average to 0.10000001 in single precision, so the group's
 * spread is 0 only when it comes from the values themselves. The other group is
 * 10, 12: `spread` gives (0.1 + 11) / 2 and `standard_error` (0.1 * 2 + 11 * 7) / 9.
 */
TEST(CutpointBetweenCenters, ConstantGroupHasNoSpreadDespiteRounding) {
  EXPECT_FLOAT_EQ(cutpoint_with_rounded_mean(Weight::Spread), 5.55F);
  EXPECT_FLOAT_EQ(cutpoint_with_rounded_mean(Weight::StandardError), 77.2F / 9.0F);
}

/* Swapping the group labels leaves every rule's cutpoint where it was. */
TEST(CutpointBetweenCenters, CutpointDoesNotDependOnGroupLabels) {
  FeatureMatrix const x = MAT(Feature, rows(7), 0, 1, 2, 9, 10, 12, 20);

  for (Center const center : {Center::Mean, Center::Median}) {
    for (Weight const weight : {Weight::None, Weight::Size, Weight::Spread, Weight::StandardError}) {
      NodeContextFixture f(x, VEC(GroupId, 1, 1, 1, 1, 0, 0, 0));
      f.ctx.projector = VEC(Feature, 1);
      BetweenCenters(center, weight).locate(f.ctx, f.rng);

      EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), cutpoint_of(center, weight));
    }
  }
}
