/**
 * @file Predict.cpp
 * @brief Predict subcommand handler.
 */
#include "cli/Predict.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <optional>

#include "io/Color.hpp"
#include "io/IO.hpp"
#include "io/Output.hpp"
#include "io/Presentation.hpp"
#include "serialization/Json.hpp"
#include "stats/Metrics.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::io::style;
using json = nlohmann::json;

namespace ppforest2::cli {
  void setup_predict(CLI::App& app, Params& params) {
    auto* sub = app.add_subcommand("predict", "Load a model and predict on new data");
    sub->add_option("-M,--model", params.model_path, "Saved model JSON file");
    sub->add_option("-d,--data", params.data_path, "CSV data to predict on, with or without a response column");
    sub->add_option("-o,--output", params.output_path, "Save prediction results to JSON file");
    sub->add_flag("--no-metrics", params.no_metrics, "Don't read the response column; omit metrics from output");
    sub->add_flag("--no-proportions", params.no_proportions, "Omit vote proportions from output");

    // CLI-exclusive constraints (predict doesn't use config files)
    sub->get_option("--model")->required()->check(CLI::ExistingFile);
    sub->get_option("--data")->required()->check(CLI::ExistingFile);

    sub->callback([&]() { params.subcommand = Subcommand::predict; });
  }

  namespace {
    std::string model_display_name(json const& model_data) {
      return model_data.value("model_type", "tree") == "forest" ? "Random Forest" : "Decision Tree";
    }

    // Parses the response column into the model's outcome space. Its error
    // names `--no-metrics`, since the column read as the response may not be
    // one at all (an ID column after the features, say).
    OutcomeVector read_response(io::csv::ResponseColumn const& response, Mode mode, Names const& groups) {
      try {
        return io::csv::parse_response(response, mode, groups);
      } catch (UserError const& e) {
        throw UserError(fmt::format("{} Use --no-metrics to predict without reading the response column.", e.what()));
      }
    }
  }

  int run_predict(Params const& params) {
    io::Output out(params.quiet);

    if (!params.output_path.empty()) {
      io::check_file_not_exists(params.output_path);
    }

    json const model_data = io::json::read_file(params.model_path, user_error);
    auto const exported   = model_data.get<serialization::Export<Model::Ptr>>();
    Mode const mode       = exported.spec->mode;

    // Columns are matched to the model's features as `serve` matches a
    // request: by name, or by position for a model without feature names.
    // The data may or may not carry a response column. Rows keep their file
    // order so `predictions[i]` corresponds to input row i.
    auto const data = io::csv::read_prediction_data(params.data_path, exported.feature_names, exported.n_features);

    auto const& model      = *exported.model;
    auto const predictions = model.predict(data.x);

    // `exported.groups` is the canonical training-label space — non-empty for
    // classification (validate_*_export enforces this), empty for regression.
    auto const& group_names = exported.groups;

    bool const has_metrics     = data.response && !params.no_metrics;
    bool const has_output_file = !params.output_path.empty();

    json result;
    if (has_output_file) {
      result["predictions"] = serialization::to_json(predictions, group_names);
    }

    if (has_output_file && is_classification(mode) && !params.no_proportions) {
      result["proportions"] = predict_proportions(model, data.x);
    }

    if (has_metrics) {
      // Labels are looked up in the model's label space, so the file may
      // list classes in any order; one the model wasn't trained on fails.
      OutcomeVector const y = read_response(*data.response, mode, group_names);
      Metrics const metrics = metrics_from_outcomes(predictions, y, mode);

      out.println("{}", emphasis("Prediction results for " + model_display_name(model_data)));
      out.newline();
      io::print_metrics_block(out, metrics, group_names);

      if (has_output_file) {
        result["metrics"] = serialization::to_json(metrics, group_names);
      }
    } else if (!has_output_file) {
      // No metrics and no file: report what was predicted rather than
      // printing nothing at all.
      auto const n = predictions.size();
      out.println("{}", emphasis("Prediction results for " + model_display_name(model_data)));
      out.newline();
      out.println(
          "Predicted {} observation{}.{}",
          n,
          n == 1 ? "" : "s",
          data.response ? "" : " The data has no response column, so there are no metrics."
      );
      out.newline();
    }

    if (!has_output_file) {
      out.println("{}", muted("Tip: use --output <file> to save individual predictions"));
    }

    if (has_output_file) {
      io::json::write_file(result, params.output_path, user_error);
      out.saved("Results", params.output_path);
    }

    return 0;
  }
}
