#include <gtest/gtest.h>

#include "stats/Entropy.hpp"

#include <cmath>

using namespace ppforest2::stats;

TEST(Entropy, SingleGroupHasNoEntropy) {
  EXPECT_DOUBLE_EQ(entropy({7}), 0.0);
}

TEST(Entropy, EqualGroupsReachLogOfTheirCount) {
  EXPECT_DOUBLE_EQ(entropy({5, 5}), std::log(2.0));
  EXPECT_DOUBLE_EQ(entropy({3, 3, 3}), std::log(3.0));
}

/* -(0.25 * log(0.25) + 0.75 * log(0.75)). */
TEST(Entropy, UnequalGroups) {
  EXPECT_NEAR(entropy({1, 3}), 0.5623351446188083, 1e-12);
}

TEST(Entropy, EmptyGroupsDoNotCount) {
  EXPECT_DOUBLE_EQ(entropy({4, 0, 4}), entropy({4, 4}));
}

TEST(Entropy, NoObservationsHasNoEntropy) {
  EXPECT_DOUBLE_EQ(entropy({}), 0.0);
  EXPECT_DOUBLE_EQ(entropy({0, 0}), 0.0);
}
