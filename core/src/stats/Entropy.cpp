#include "stats/Entropy.hpp"

#include <cmath>
#include <numeric>

namespace ppforest2::stats {
  double entropy(std::vector<int> const& group_sizes) {
    int const total = std::accumulate(group_sizes.begin(), group_sizes.end(), 0);

    if (total == 0) {
      return 0;
    }

    double result = 0;

    for (int const size : group_sizes) {
      if (size > 0) {
        double const share = static_cast<double>(size) / total;
        result -= share * std::log(share);
      }
    }

    return result;
  }
}
