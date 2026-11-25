#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/binarize/NearestGroups.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/binarize/LargestGap.hpp"
#include "models/strategies/NodeContext.hpp"
#include "models/strategies/pp/PDA.hpp"
#include "models/strategies/vars/All.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <cmath>
#include <set>

using namespace ppforest2;
using namespace ppforest2::binarize;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

using Gap = NearestGroups::Gap;

TEST(BinarizeNearestGroups, FromJsonRoundTrip) {
  json const j = {{"name", "nearest_groups"}, {"gap", "min_entropy"}};

  EXPECT_EQ(NearestGroups::from_json(j)->to_json(), j);
}

TEST(BinarizeNearestGroups, FromJsonDefaultsToTheLargestGap) {
  json const j        = {{"name", "nearest_groups"}};
  json const expected = {{"name", "nearest_groups"}, {"gap", "largest"}};

  EXPECT_EQ(NearestGroups::from_json(j)->to_json(), expected);
}

TEST(BinarizeNearestGroups, FromJsonUnknownGapThrows) {
  json const j = {{"name", "nearest_groups"}, {"gap", "smallest"}};
  EXPECT_THROW(NearestGroups::from_json(j), std::runtime_error);
}

TEST(BinarizeNearestGroups, FromJsonUnknownParamThrows) {
  json const j = {{"name", "nearest_groups"}, {"extra", 1}};
  EXPECT_THROW(NearestGroups::from_json(j), std::runtime_error);
}

TEST(BinarizeNearestGroups, RegistryLookup) {
  json const j = {{"name", "nearest_groups"}, {"gap", "largest"}};

  EXPECT_EQ(Binarization::from_json(j)->to_json(), j);
}

TEST(BinarizeNearestGroups, DisplayNameNamesTheGap) {
  EXPECT_EQ(NearestGroups(Gap::MinEntropy).display_name(), "Nearest groups (gap: min_entropy)");
}

/*
 * Projected means 1, 2 and 10: the largest gap is between groups 1 and 2,
 * so the sides are {0, 1} and {2}, as with LargestGap, and the projection
 * is left to groups 1 and 2, the ones next to the gap.
 */
TEST(BinarizeNearestGroups, KeepsTheTwoGroupsNextToTheLargestGap) {
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 1, 2, 2, 10, 10);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);
  NodeContextFixture g(x, y);
  f.ctx.projector = VEC(Feature, 1);
  g.ctx.projector = VEC(Feature, 1);

  NearestGroups(Gap::Largest).regroup(f.ctx, f.rng);
  LargestGap().regroup(g.ctx, g.rng);

  auto const& sides = f.ctx.y_bin.value();
  EXPECT_EQ(sides.subgroups, g.ctx.y_bin.value().subgroups);
  EXPECT_EQ(sides.subgroups.at(0), (std::set<GroupId>{0, 1}));
  EXPECT_EQ(sides.subgroups.at(1), (std::set<GroupId>{2}));

  auto const& nearest = f.ctx.projection_partition();
  EXPECT_EQ(nearest.groups, (std::set<GroupId>{1, 2}));
  EXPECT_EQ(nearest.total_size(), 4);

  EXPECT_EQ(f.ctx.active_partition().total_size(), 6);
  EXPECT_EQ(g.ctx.projection_partition().total_size(), 6);
}

/*
 * Groups 0 (0, 0, 0), 1 (4, 4, 4) and 2 (3, 3, 24), with means 0, 4 and 10.
 * The largest gap is between groups 1 and 2, but its midpoint, 7, leaves
 * eight mixed observations below it. The midpoint of the other gap, 2,
 * has a pure side, so the entropy rule cuts there: {0} against {1, 2}.
 */
TEST(BinarizeNearestGroups, MinEntropyGapCanDifferFromTheLargest) {
  FeatureMatrix const x = MAT(Feature, rows(9), 0, 0, 0, 4, 4, 4, 3, 3, 24);
  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 1, 1, 1, 2, 2, 2);

  NodeContextFixture largest(x, y);
  NodeContextFixture entropy(x, y);
  largest.ctx.projector = VEC(Feature, 1);
  entropy.ctx.projector = VEC(Feature, 1);

  NearestGroups(Gap::Largest).regroup(largest.ctx, largest.rng);
  NearestGroups(Gap::MinEntropy).regroup(entropy.ctx, entropy.rng);

  EXPECT_EQ(largest.ctx.y_bin.value().subgroups.at(0), (std::set<GroupId>{0, 1}));
  EXPECT_EQ(largest.ctx.projection_partition().groups, (std::set<GroupId>{1, 2}));

  EXPECT_EQ(entropy.ctx.y_bin.value().subgroups.at(0), (std::set<GroupId>{0}));
  EXPECT_EQ(entropy.ctx.projection_partition().groups, (std::set<GroupId>{0, 1}));
}

/* With equal means there is no gap to cut at: one side is empty and the node is aborted. */
TEST(BinarizeNearestGroups, AbortsWithoutAGap) {
  Feature const nan = std::nanf("");

  NodeContextFixture f(MAT(Feature, rows(6), nan, nan, nan, nan, nan, nan), VEC(GroupId, 0, 0, 1, 1, 2, 2));
  f.ctx.projector = VEC(Feature, 1);

  NearestGroups(Gap::Largest).regroup(f.ctx, f.rng);

  EXPECT_TRUE(f.ctx.aborted);
  EXPECT_FALSE(f.ctx.y_projection.has_value());
}

/*
 * The projection after the binarization is the one of the two nearest
 * groups alone: it equals the projection fitted on data that holds only
 * those two groups, whatever the third one looks like.
 */
TEST(BinarizeNearestGroups, ProjectionIsFittedOnTheNearestGroups) {
  FeatureMatrix const x =
      MAT(Feature, rows(12), 0, 0, 1, 2, 2, 1, 3, 3, 1, 6, 2, 8, 3, 7, 0, 9, 50, 12, 90, 13, 70, 15, 60, 14);
  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2);

  NodeContextFixture all(x, y);
  all.ctx.var_selection = vars::VariableSelection::Result({0, 1}, 2);
  all.ctx.projector     = VEC(Feature, 0, 1);

  NearestGroups(Gap::Largest).regroup(all.ctx, all.rng);
  ASSERT_EQ(all.ctx.projection_partition().groups, (std::set<GroupId>{0, 1}));

  pp::PDA(0).optimize(all.ctx, all.rng);

  NodeContextFixture pair(FeatureMatrix(x.topRows(8)), GroupIdVector(y.head(8)));
  pair.ctx.var_selection = vars::VariableSelection::Result({0, 1}, 2);

  pp::PDA(0).optimize(pair.ctx, pair.rng);

  Projector const fitted   = all.ctx.projector.value();
  Projector const expected = pair.ctx.projector.value();

  EXPECT_NEAR(std::abs(fitted.normalized().dot(expected.normalized())), 1.0F, 1e-4F);
}
