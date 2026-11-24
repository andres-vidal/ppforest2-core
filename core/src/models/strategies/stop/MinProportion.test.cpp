#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/MinProportion.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "stats/GroupPartition.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"
#include "utils/UserError.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;
using namespace ppforest2::stop;

namespace {
  // A two-group node holding the first `node_size` of 100 training observations.
  bool stops(int node_size, Feature proportion) {
    NodeContextFixture f(
        FeatureMatrix::Zero(100, 1),
        GroupPartition(0, node_size - 1).bisect(node_size / 2),
        OutcomeVector(OutcomeVector::Zero(100))
    );

    return MinProportion(proportion).should_stop(f.ctx, f.rng);
  }
}

TEST(MinProportionStop, FromJsonRoundTrip) {
  json const j = {{"name", "min_proportion"}, {"proportion", 0.05F}};

  EXPECT_EQ(MinProportion::from_json(j)->to_json(), j);
}

TEST(MinProportionStop, FromJsonUnknownParamThrows) {
  json const j = {{"name", "min_proportion"}, {"proportion", 0.05F}, {"extra", 1}};
  EXPECT_THROW(MinProportion::from_json(j), std::runtime_error);
}

TEST(MinProportionStop, RegistryLookup) {
  json const j = {{"name", "min_proportion"}, {"proportion", 0.05F}};

  EXPECT_EQ(StopRule::from_json(j)->to_json(), j);
}

TEST(MinProportionStop, ProportionMustBeBetweenZeroAndOne) {
  EXPECT_THROW(MinProportion(0.0F), UserError);
  EXPECT_THROW(MinProportion(1.0F), UserError);
  EXPECT_THROW(MinProportion(-0.2F), UserError);
}

TEST(MinProportionStop, DisplayNameShowsTheProportion) {
  EXPECT_EQ(MinProportion(0.05F).display_name(), "Min proportion (0.05)");
}

TEST(MinProportionStop, StopsANodeAtOrBelowTheProportion) {
  EXPECT_TRUE(stops(4, 0.05F));
  EXPECT_TRUE(stops(5, 0.05F));
}

TEST(MinProportionStop, KeepsSplittingALargerNode) {
  EXPECT_FALSE(stops(6, 0.05F));
  EXPECT_FALSE(stops(100, 0.05F));
}
