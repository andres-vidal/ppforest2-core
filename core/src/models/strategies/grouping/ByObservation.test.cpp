#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/grouping/ByObservation.hpp"
#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;
using namespace ppforest2::grouping;

TEST(GroupingByObservation, FromJsonRoundTrip) {
  json const j = {{"name", "by_observation"}};

  EXPECT_EQ(ByObservation::from_json(j)->to_json(), j);
}

TEST(GroupingByObservation, FromJsonUnknownParamThrows) {
  json const j = {{"name", "by_observation"}, {"extra", 1}};
  EXPECT_THROW(ByObservation::from_json(j), std::runtime_error);
}

TEST(GroupingByObservation, RegistryLookup) {
  json const j = {{"name", "by_observation"}};

  EXPECT_EQ(Grouping::from_json(j)->to_json(), j);
}

TEST(GroupingByObservation, DoesNotRequireTwoGroups) {
  EXPECT_FALSE(ByObservation().requires_two_groups());
}

TEST(GroupingByObservation, InitHasOneGroupPerLabel) {
  auto const partition = ByObservation().init(VEC(GroupId, 0, 0, 1, 2, 2));

  EXPECT_EQ(partition.groups.size(), 3U);
  EXPECT_EQ(partition.group_size(0), 2);
  EXPECT_EQ(partition.group_size(1), 1);
  EXPECT_EQ(partition.group_size(2), 2);
}

/*
 * Group 0 sits on both sides of group 1 (0, 1 and 10, 11 around 5, 6). A
 * cutpoint of 8 sends two of its observations to each child.
 */
TEST(GroupingByObservation, DividesAGroupBetweenTheChildren) {
  NodeContextFixture f(MAT(Feature, rows(6), 0, 10, 1, 11, 5, 6), VEC(GroupId, 0, 0, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);
  f.ctx.cutpoint  = 8.0F;

  ByObservation().split(f.ctx, f.rng);

  ASSERT_FALSE(f.ctx.aborted);

  auto const& lower = f.ctx.lower_y_part.value();
  auto const& upper = f.ctx.upper_y_part.value();

  EXPECT_EQ(lower.total_size(), 4);
  EXPECT_EQ(lower.group_size(0), 2);
  EXPECT_EQ(lower.group_size(1), 2);

  EXPECT_EQ(upper.total_size(), 2);
  EXPECT_EQ(upper.groups.size(), 1U);
  EXPECT_EQ(upper.group_size(0), 2);
}

/* Each child's rows are the ones on its side of the cutpoint, in their original order. */
TEST(GroupingByObservation, ReordersRowsSoEachChildOwnsItsSide) {
  NodeContextFixture f(MAT(Feature, rows(6), 0, 10, 1, 11, 5, 6), VEC(GroupId, 0, 0, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);
  f.ctx.cutpoint  = 8.0F;

  ByObservation().split(f.ctx, f.rng);

  FeatureMatrix const lower_rows = f.ctx.lower_y_part.value().data(f.x);
  FeatureMatrix const upper_rows = f.ctx.upper_y_part.value().data(f.x);

  EXPECT_EQ(lower_rows, MAT(Feature, rows(4), 0, 1, 5, 6));
  EXPECT_EQ(upper_rows, MAT(Feature, rows(2), 10, 11));
  EXPECT_EQ(f.y_vec, VEC(Outcome, 0, 0, 0, 0, 1, 1));
}

/* A value equal to the cutpoint goes to the upper child, as in prediction. */
TEST(GroupingByObservation, ValueOnTheCutpointGoesUp) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);
  f.ctx.cutpoint  = 2.0F;

  ByObservation().split(f.ctx, f.rng);

  EXPECT_EQ(f.ctx.lower_y_part.value().total_size(), 1);
  EXPECT_EQ(f.ctx.upper_y_part.value().total_size(), 3);
}

/* With every observation on one side there is no split, and the node is aborted. */
TEST(GroupingByObservation, AbortsWhenEveryObservationIsOnOneSide) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1);
  f.ctx.cutpoint  = 100.0F;

  ByObservation().split(f.ctx, f.rng);

  EXPECT_TRUE(f.ctx.aborted);
}
