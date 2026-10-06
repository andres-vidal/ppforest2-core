#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "test/Projectors.hpp"
#include "utils/UserError.hpp"

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::test;
using namespace ppforest2::types;

TEST(TreeBranch, EqualsEqualConditions) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeBranch, EqualsCollinearProjectors) {
  TreeBranch const c1(as_projector({1.0, 1.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({2.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeBranch, EqualsApproximateThresholds) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 3.000000000000001, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeBranch, EqualsNonCollinearProjectors) {
  TreeBranch const c1(as_projector({1.0, 0.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({0.0, 1.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeBranch, EqualsDifferentThresholds) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 4.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeBranch, EqualsDifferentResponses) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeBranch, EqualsDifferentStructures) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(
      as_projector({1.0, 2.0}),
      3.0,
      TreeLeaf::make(1),
      TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2))
  );

  ASSERT_FALSE(c1 == c2);
}

TEST(IsLeaf, FalseForTreeBranch) {
  TreeBranch const branch(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  EXPECT_FALSE(is_leaf(branch));
}

TEST(IsLeaf, FalseForTreeBranchPtr) {
  TreeNode::Ptr const branch = TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  EXPECT_FALSE(is_leaf(branch));
}

TEST(TreeBranch, PredictRejectsWrongFeatureCount) {
  TreeBranch const branch(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  FeatureVector too_many(3);
  too_many << 1.0F, 2.0F, 3.0F;
  FeatureVector too_few(1);
  too_few << 1.0F;

  EXPECT_THROW(branch.predict(too_many), UserError);
  EXPECT_THROW(branch.predict(too_few), UserError);
}

TEST(TreeBranch, PredictAcceptsMatchingFeatureCount) {
  TreeBranch const branch(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  FeatureVector x(2);
  x << 1.0F, 2.0F;

  EXPECT_EQ(branch.predict(x), 2);
}
