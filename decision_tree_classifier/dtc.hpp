#ifndef decision_tree_classifier_HPP
#define decision_tree_classifier_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>

template <typename T>
class decision_tree_classifier{
  private:
    struct Node{
      bool is_leaf;
      T threshold;
      int label;
      int feature;
      std::unique_ptr<Node> left;
      std::unique_ptr<Node> right;
    
      Node(bool leaf = false, int lbl = -1, int f = -1, T thresh = 0.0)
        : is_leaf(leaf), label(lbl), feature(f), threshold(thresh), \
         left(nullptr), right(nullptr) {}
    };

    std::unique_ptr<Node> root;
    int max_depth;

    // Helper function to calculate the entropy for a given set of labels
    T entropy(const std::vector<int>& labels){
      std::map<int, int> counts;
      for (int lbl : labels){
        counts[lbl]++;
      }
      T entropy = 0.0;
      for (const auto& pair : counts){
        T prob = static_cast<T>(pair.second) / labels.size();
        entropy -= prob * std::log2(prob + 1e-10);
      }
      return entropy;
    }

    // Find best threshold for a single feature
    std::pair<double, double> find_best_threshold(const std::vector<double>& feature, \
      const std::vector<int>& labels) {
      std::vector<std::pair<double, int>> sorted_data;
      for (size_t i = 0; i < feature.size(); ++i) {
          sorted_data.emplace_back(feature[i], labels[i]);
      }
      std::sort(sorted_data.begin(), sorted_data.end());

      double best_gain = 0.0;
      double best_threshold = 0.0;
      double parent_entropy = entropy(labels);

      for (size_t i = 1; i < sorted_data.size(); ++i) {
          if (sorted_data[i].first == sorted_data[i-1].first) continue;

          double threshold = (sorted_data[i].first + sorted_data[i-1].first) / 2.0;
          std::vector<int> left_labels, right_labels;
          for (size_t j = 0; j < i; ++j) left_labels.push_back(sorted_data[j].second);
          for (size_t j = i; j < sorted_data.size(); ++j) right_labels.push_back(sorted_data[j].second);

          double curr_entropy = (left_labels.size() * entropy(left_labels) + 
                                right_labels.size() * entropy(right_labels));
          curr_entropy /= labels.size();
          double gain = parent_entropy - curr_entropy;

          if (gain > best_gain) {
              best_gain = gain;
              best_threshold = threshold;
          }
      }
      return {best_threshold, best_gain};
  }

  // Find best split across all features
  std::tuple<int, double, double> find_best_split(const std::vector<std::vector<double>>& data, 
                                                 const std::vector<int>& labels) {
      int best_feature = -1;
      double best_threshold = 0.0;
      double best_gain = 0.0;

      for (size_t feature_idx = 0; feature_idx < data[0].size(); ++feature_idx) {
          std::vector<double> feature_col;
          for (const auto& row : data) feature_col.push_back(row[feature_idx]);

          auto [threshold, gain] = find_best_threshold(feature_col, labels);
          if (gain > best_gain) {
              best_gain = gain;
              best_threshold = threshold;
              best_feature = feature_idx;
          }
      }
      return {best_feature, best_threshold, best_gain};
  }

  // Recursive tree-building function
  std::unique_ptr<Node> build_tree(const std::vector<std::vector<double>>& data, 
                                  const std::vector<int>& labels, 
                                  int depth = 0) {
      // Check stopping conditions
      std::vector<int> unique_labels(labels.begin(), labels.end());
      std::sort(unique_labels.begin(), unique_labels.end());
      unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());

      if (unique_labels.size() == 1) {
          return std::make_unique<Node>(true, labels[0]);
      }

      if (max_depth != -1 && depth >= max_depth) {
          std::map<int, int> counts;
          for (int lbl : labels) counts[lbl]++;
          int majority = std::max_element(counts.begin(), counts.end(), 
              [](auto& a, auto& b) { return a.second < b.second; })->first;
          return std::make_unique<Node>(true, majority);
      }

      auto [feature, threshold, gain] = find_best_split(data, labels);
      if (feature == -1 || gain <= 0.0) {
          std::map<int, int> counts;
          for (int lbl : labels) counts[lbl]++;
          int majority = std::max_element(counts.begin(), counts.end(), 
              [](auto& a, auto& b) { return a.second < b.second; })->first;
          return std::make_unique<Node>(true, majority);
      }

      auto node = std::make_unique<Node>(false, -1, feature, threshold);
      std::vector<std::vector<double>> left_data, right_data;
      std::vector<int> left_labels, right_labels;

      for (size_t i = 0; i < data.size(); ++i) {
          if (data[i][feature] <= threshold) {
              left_data.push_back(data[i]);
              left_labels.push_back(labels[i]);
          } else {
              right_data.push_back(data[i]);
              right_labels.push_back(labels[i]);
          }
      }

      node->left = build_tree(left_data, left_labels, depth + 1);
      node->right = build_tree(right_data, right_labels, depth + 1);
      return node;
  }

  public:
    decision_tree_classifier(int max_depth = -1) : max_depth(max_depth) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        root = build_tree(X, y);
    }

    int predict_sample(const std::vector<double>& x, const std::unique_ptr<Node>& node) const {
        if (node->is_leaf) return node->label;
        if (x[node->feature] <= node->threshold) {
            return predict_sample(x, node->left);
        } else {
            return predict_sample(x, node->right);
        }
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<int> predictions;
        for (const auto& x : X) {
            predictions.push_back(predict_sample(x, root));
        }
        return predictions;
    }
};
#endif // decision_tree_classifier_HPP