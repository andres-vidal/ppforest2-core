#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/cutpoint/MinEntropy.hpp"
#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <cmath>

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;
using namespace ppforest2::cutpoint;

TEST(CutpointMinEntropy, FromJsonRoundTrip) {
  json const j = {{"name", "min_entropy"}};

  EXPECT_EQ(MinEntropy::from_json(j)->to_json(), j);
}

TEST(CutpointMinEntropy, FromJsonUnknownParamThrows) {
  json const j = {{"name", "min_entropy"}, {"extra", 1}};
  EXPECT_THROW(MinEntropy::from_json(j), std::runtime_error);
}

TEST(CutpointMinEntropy, RegistryLookup) {
  json const j = {{"name", "min_entropy"}};

  EXPECT_EQ(Cutpoint::from_json(j)->to_json(), j);
}

TEST(CutpointMinEntropy, DoesNotRequireTwoGroups) {
  EXPECT_FALSE(MinEntropy().requires_two_groups());
}

/* Two separated groups: the only cut with two pure sides is between 2 and 5. */
TEST(CutpointMinEntropy, SeparatedGroupsAreCutBetweenThem) {
  NodeContextFixture f(MAT(Feature, rows(5), 0, 1, 2, 5, 9), VEC(GroupId, 0, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 3.5F);
  EXPECT_EQ(f.ctx.lower_group.value(), 0);
  EXPECT_EQ(f.ctx.upper_group.value(), 1);
}

/* The mean of means of 0, 1, 2 | 3, 30 is 9; the entropy cut stays at the boundary. */
TEST(CutpointMinEntropy, FollowsTheBoundaryNotTheMeans) {
  NodeContextFixture f(MAT(Feature, rows(5), 0, 1, 2, 3, 30), VEC(GroupId, 0, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 2.5F);
}

/*
 * Overlapping groups, projected values 0 0 1 | 1.5(g1) | 2(g0) | 3 4 (g1):
 * cutting at 1.25 leaves {3 g0} and {1 g0, 3 g1}, score 4/7 * E(1, 3) = 0.3213;
 * cutting at 2.5 leaves {4 g0, 1 g1} and {2 g1}, score 5/7 * E(4, 1) = 0.3574.
 */
TEST(CutpointMinEntropy, OverlappingGroupsTakeTheLowestWeightedEntropy) {
  NodeContextFixture f(MAT(Feature, rows(7), 0, 0, 1, 2, 1.5, 3, 4), VEC(GroupId, 0, 0, 0, 0, 1, 1, 1));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 1.25F);
}

/* Three groups, no binarization: the purest cut isolates group 0. */
TEST(CutpointMinEntropy, WorksOnMoreThanTwoGroups) {
  NodeContextFixture f(MAT(Feature, rows(7), 0, 1, 2, 10, 11, 12, 13), VEC(GroupId, 0, 0, 0, 1, 1, 2, 2));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 6.0F);
  EXPECT_FALSE(f.ctx.lower_group.has_value());
  EXPECT_FALSE(f.ctx.upper_group.has_value());
}

/* 0 0 | 1 1 | 0 0 on a line: both cuts score the same, and the lowest wins. */
TEST(CutpointMinEntropy, EqualScoresTakeTheLowestCutpoint) {
  NodeContextFixture f(MAT(Feature, rows(6), 0, 1, 4, 5, 2, 3), VEC(GroupId, 0, 0, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 1.5F);
}

/* Equal projected values are never separated: the cut is between 1 and 2. */
TEST(CutpointMinEntropy, TiedValuesStayTogether) {
  NodeContextFixture f(MAT(Feature, rows(5), 1, 1, 1, 1, 2), VEC(GroupId, 0, 0, 1, 1, 1));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 1.5F);
}

/* The cutpoint between two adjacent floats is the upper one, so the lower is below it. */
TEST(CutpointMinEntropy, AdjacentValuesAreStillSeparated) {
  Feature const lower = 1.0F;
  Feature const upper = std::nextafter(lower, 2.0F);

  NodeContextFixture f(MAT(Feature, rows(2), lower, upper), VEC(GroupId, 0, 1));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_GT(f.ctx.cutpoint.value(), lower);
  EXPECT_LE(f.ctx.cutpoint.value(), upper);
}

TEST(CutpointMinEntropy, AbortsWhenEveryProjectedValueIsEqual) {
  NodeContextFixture f(MAT(Feature, rows(4), 3, 3, 3, 3), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);

  MinEntropy().locate(f.ctx, f.rng);

  EXPECT_TRUE(f.ctx.aborted);
  EXPECT_FALSE(f.ctx.cutpoint.has_value());
}
