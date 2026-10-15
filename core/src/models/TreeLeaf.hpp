#pragma once

#include "models/TreeNode.hpp"

namespace ppforest2 {
  /**
   * @brief Leaf node in a projection pursuit tree.
   *
   * Holds a single class label and always returns it as the prediction,
   * regardless of the input feature vector.
   */
  class TreeLeaf : public TreeNode {
  public:
    using Ptr = std::unique_ptr<TreeLeaf>;

    /** @brief Class label stored at this leaf. */
    types::Outcome value;

    explicit TreeLeaf(types::Outcome value);

    void accept(TreeNode::Visitor& visitor) const override;

    /**
     * @brief Return the stored class label (ignores input).
     *
     * @param x  Feature vector (unused).
     * @return   The class label stored at this leaf.
     */
    types::Outcome predict(types::FeatureVector const& x) const override;

    bool equals(TreeNode const& other) const override;

    /** @brief Factory method that returns a unique_ptr to a new TreeLeaf. */
    static Ptr make(types::Outcome value);
  };
}
