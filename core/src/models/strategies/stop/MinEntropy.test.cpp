#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/MinEntropy.hpp"
#include "models/strategies/stop/StopRule.hpp"
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
  // A node with 9 observations of group 0 and `minority` of group 1.
  bool stops(int minority, Feature threshold) {
    GroupIdVector ids = GroupIdVector::Zero(9 + minority);
    ids.tail(minority).setOnes();

    NodeContextFixture f(FeatureMatrix::Zero(9 + minority, 1), ids);

    return MinEntropy(threshold).should_stop(f.ctx, f.rng);
  }
}

TEST(MinEntropyStop, FromJsonRoundTrip) {
  json const j = {{"name", "min_entropy"}, {"threshold", 0.5F}};

  EXPECT_EQ(MinEntropy::from_json(j)->to_json(), j);
}

TEST(MinEntropyStop, FromJsonUnknownParamThrows) {
  json const j = {{"name", "min_entropy"}, {"threshold", 0.5F}, {"extra", 1}};
  EXPECT_THROW(MinEntropy::from_json(j), std::runtime_error);
}

TEST(MinEntropyStop, FromJsonNegativeThresholdThrows) {
  json const j = {{"name", "min_entropy"}, {"threshold", -0.1F}};
  EXPECT_THROW(MinEntropy::from_json(j), std::runtime_error);
}

TEST(MinEntropyStop, RegistryLookup) {
  json const j = {{"name", "min_entropy"}, {"threshold", 0.5F}};

  EXPECT_EQ(StopRule::from_json(j)->to_json(), j);
}

TEST(MinEntropyStop, NegativeThresholdIsAUserError) {
  EXPECT_THROW(MinEntropy(-1.0F), UserError);
}

TEST(MinEntropyStop, DisplayNameShowsTheThreshold) {
  EXPECT_EQ(MinEntropy(0.5F).display_name(), "Min entropy (0.5)");
}

/* 9 against 1 has entropy 0.325, below 0.5: almost pure, so it stops. */
TEST(MinEntropyStop, StopsAnAlmostPureNode) {
  EXPECT_TRUE(stops(1, 0.5F));
}

/* 9 against 9 has entropy log(2) = 0.693, above 0.5: it keeps splitting. */
TEST(MinEntropyStop, KeepsSplittingAMixedNode) {
  EXPECT_FALSE(stops(9, 0.5F));
}

/* A threshold of 0 never stops a node: no entropy is below it. */
TEST(MinEntropyStop, ZeroThresholdNeverStops) {
  EXPECT_FALSE(stops(1, 0.0F));
}
