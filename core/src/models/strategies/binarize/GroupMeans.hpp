#pragma once

#include "stats/GroupPartition.hpp"
#include "utils/Types.hpp"

#include <cstddef>
#include <vector>

namespace ppforest2 {
  struct NodeContext;
}

/**
 * @brief What the binarizations that cut between projected group means share.
 *
 * They order the groups of a node by projected mean, pick one of the gaps
 * between consecutive means, and relabel the groups below it as 0 and the
 * groups above it as 1.
 */
namespace ppforest2::binarize {
  /** @brief A group and the mean of its projected observations. */
  struct GroupMean {
    types::GroupId group;
    types::Feature mean;
  };

  /**
   * @brief The node's groups with their projected means, in ascending order
   *        of mean. Groups with equal means keep their label order.
   */
  std::vector<GroupMean> sorted_group_means(NodeContext const& ctx, types::FeatureVector const& projected);

  /**
   * @brief Position in @p means of the first group above the largest gap
   *        between consecutive means; the first such gap when several are
   *        equal. 0, which puts every group on one side, when there is no
   *        gap to compare (a mean is not a number).
   */
  std::size_t largest_gap_edge(std::vector<GroupMean> const& means);

  /**
   * @brief The 2-group partition of @p y with the groups before position
   *        @p edge of @p means as group 0 and the rest as group 1.
   */
  stats::GroupPartition
  binarize_at(stats::GroupPartition const& y, std::vector<GroupMean> const& means, std::size_t edge);
}
