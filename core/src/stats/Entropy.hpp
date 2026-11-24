#pragma once

#include <vector>

namespace ppforest2::stats {
  /**
   * @brief Shannon entropy of a set of observations, from its group sizes.
   *
   * `-sum(p_g * log(p_g))` with `p_g` the share of group g and the natural
   * logarithm. It is 0 for a single group and `log(G)` for G groups of
   * equal size; empty groups do not count.
   */
  double entropy(std::vector<int> const& group_sizes);
}
