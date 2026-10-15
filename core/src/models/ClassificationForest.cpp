#include "models/ClassificationForest.hpp"

#include "models/Bagged.hpp"
#include "models/ClassificationTree.hpp"
#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"
#include "utils/Invariant.hpp"

#include <Eigen/Dense>
#include <algorithm>

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  using GroupIndices = ClassificationForest::GroupIndices;
  using Ptr          = ClassificationForest::Ptr;

  namespace {
    /**
     * @brief Stratified per-group sample: draw `group_size(g)` indices
     *        uniformly from each group `g` in the training data.
     *
     * Preserves the class balance across bootstrap replicates.
     */
    std::vector<int> stratified_sample(GroupPartition const& y_part, RNG& rng) {
      std::vector<int> indices;
      indices.reserve(static_cast<std::size_t>(y_part.total_size()));

      for (auto const& group : y_part.groups) {
        Uniform const unif(y_part.group_start(group), y_part.group_end(group));
        std::vector<int> const sample = unif(y_part.group_size(group), rng);
        indices.insert(indices.end(), sample.begin(), sample.end());
      }

      std::sort(indices.begin(), indices.end());
      return indices;
    }
  }


  Ptr ClassificationForest::train(TrainingSpec const& spec, FeatureMatrix const& x, OutcomeVector const& y) {
    invariant(is_classification(spec), "ClassificationForest::train requires mode = Classification");

    auto spec_ptr               = TrainingSpec::make(spec);
    GroupPartition const y_part = spec_ptr->init_groups(y);

    auto forest = std::make_unique<ClassificationForest>(spec_ptr, y_part.groups);

    forest->build_trees([&](RNG& rng) {
      std::vector<int> sample_indices = stratified_sample(y_part, rng);
      FeatureMatrix sampled_x         = x(sample_indices, Eigen::indexing::all);
      OutcomeVector sampled_y         = y(sample_indices);

      auto tree = ClassificationTree::train(*spec_ptr, sampled_x, sampled_y, y_part, rng);

      return BaggedTree::make(std::move(tree), std::move(sample_indices));
    });

    return forest;
  }

  Outcome ClassificationForest::predict(FeatureVector const& x) const {
    invariant(!trees.empty(), "Forest has no trees.");

    std::vector<Outcome> preds;
    preds.reserve(trees.size());

    for (auto const& tree : trees) {
      preds.push_back(tree->predict(x));
    }

    return stats::majority_vote(preds);
  }

  FeatureVector ClassificationForest::predict(FeatureVector const& x, Proportions) const {
    return predict(x, Proportions{}, group_indices(groups));
  }

  FeatureVector ClassificationForest::predict(FeatureVector const& x, Proportions, GroupIndices const& indices) const {
    invariant(!trees.empty(), "Forest has no trees.");

    int const G               = static_cast<int>(indices.size());
    FeatureVector proportions = FeatureVector::Zero(G);

    for (auto const& bag : trees) {
      auto const& tree = static_cast<ClassificationTree const&>(*bag->model);
      proportions += tree.predict(x, Proportions{}, indices);
    }

    // L1-normalize. Zero-vote case (an all-degenerate forest) leaves the
    // vector zeros — `total > 0` guards against the 0/0 → NaN trap.
    Feature const total = proportions.sum();
    if (total > 0) {
      proportions /= total;
    }

    return proportions;
  }

  FeatureMatrix ClassificationForest::predict(FeatureMatrix const& x, Proportions) const {
    return predict(x, Proportions{}, group_indices(groups));
  }

  FeatureMatrix ClassificationForest::predict(FeatureMatrix const& x, Proportions, GroupIndices const& indices) const {
    invariant(!trees.empty(), "Forest has no trees.");

    int const G = static_cast<int>(indices.size());
    int const n = static_cast<int>(x.rows());
    FeatureMatrix proportions(n, G);

    for (int i = 0; i < n; ++i) {
      proportions.row(i) = predict(static_cast<FeatureVector>(x.row(i)), Proportions{}, indices);
    }

    return proportions;
  }

  void ClassificationForest::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }
}
