#include "models/strategies/binarize/GroupMeans.hpp"

#include "models/strategies/NodeContext.hpp"

#include <algorithm>
#include <cmath>
#include <map>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::binarize {
  std::vector<GroupMean> sorted_group_means(NodeContext const& ctx, FeatureVector const& projected) {
    std::vector<GroupMean> means;
    means.reserve(ctx.y.groups.size());

    for (GroupId const group : ctx.y.groups) {
      means.push_back({group, ctx.y.group(projected, group).mean()});
    }

    std::stable_sort(means.begin(), means.end(), [](GroupMean const& a, GroupMean const& b) {
      return std::isless(a.mean, b.mean);
    });

    return means;
  }

  std::size_t largest_gap_edge(std::vector<GroupMean> const& means) {
    Feature largest  = -1;
    std::size_t edge = 0;

    for (std::size_t i = 0; i + 1 < means.size(); i++) {
      Feature const gap = means[i + 1].mean - means[i].mean;

      if (gap > largest) {
        largest = gap;
        edge    = i + 1;
      }
    }

    return edge;
  }

  GroupPartition binarize_at(GroupPartition const& y, std::vector<GroupMean> const& means, std::size_t edge) {
    std::map<GroupId, int> binary_mapping;

    for (std::size_t i = 0; i < means.size(); i++) {
      binary_mapping[means[i].group] = i < edge ? 0 : 1;
    }

    return y.remap(binary_mapping);
  }
}
