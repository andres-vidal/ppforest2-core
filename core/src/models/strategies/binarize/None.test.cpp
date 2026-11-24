#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/binarize/None.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/binarize/LargestGap.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;
using namespace ppforest2::binarize;

TEST(BinarizeNone, FromJsonRoundTrip) {
  json const j = {{"name", "none"}};

  EXPECT_EQ(None::from_json(j)->to_json(), j);
}

TEST(BinarizeNone, FromJsonUnknownParamThrows) {
  json const j = {{"name", "none"}, {"extra", 1}};
  EXPECT_THROW(None::from_json(j), std::runtime_error);
}

TEST(BinarizeNone, RegistryLookup) {
  json const j = {{"name", "none"}};

  EXPECT_EQ(Binarization::from_json(j)->to_json(), j);
}

TEST(BinarizeNone, DoesNotBinarize) {
  EXPECT_FALSE(None().binarizes());
  EXPECT_TRUE(LargestGap().binarizes());
}

/* A 3-group node keeps its 3 groups: no binary partition, no abort. */
TEST(BinarizeNone, RegroupKeepsEveryGroup) {
  NodeContextFixture f(MAT(Feature, rows(6), 0, 1, 5, 6, 10, 11), VEC(GroupId, 0, 0, 1, 1, 2, 2));
  f.ctx.projector = VEC(Feature, 1);

  None().regroup(f.ctx, f.rng);

  EXPECT_FALSE(f.ctx.y_bin.has_value());
  EXPECT_FALSE(f.ctx.aborted);
  EXPECT_EQ(f.ctx.active_partition().groups.size(), 3U);
}
