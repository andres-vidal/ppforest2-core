/**
 * @file Predict.integration.test.cpp
 * @brief Integration tests for the predict subcommand.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Predict subcommand (fixture-based)
// ---------------------------------------------------------------------------

class PredictTest : public SavedModelTest {};

/* Default predict shows error rate and confusion matrix. */
TEST_F(PredictTest, PredictWithSavedModel) {
  auto result = run_ppforest2("predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_FALSE(result.stdout_output.empty());
  EXPECT_NE(result.stdout_output.find("Prediction results for"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Error:"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics in quiet mode suppresses error rate and confusion matrix. */
TEST_F(PredictTest, PredictNoMetrics) {
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Error:"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics without quiet still suppresses metrics output. */
TEST_F(PredictTest, PredictNoMetricsWithoutQuiet) {
  // --no-metrics without -q should also suppress metrics
  auto result = run_ppforest2("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Error:"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics omits the metrics block from JSON output. */
TEST_F(PredictTest, PredictNoMetricsOutputFile) {
  TempFile output;
  output.clear();
  auto result =
      run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_FALSE(j.contains("metrics"));
}

/* Without --output, predict shows a hint about --output. */
TEST_F(PredictTest, PredictSuggestsOutputHint) {
  auto result = run_ppforest2("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("--output"), std::string::npos);
}

/* When --output is used, the hint is suppressed. */
TEST_F(PredictTest, PredictNoHintWhenOutputUsed) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Tip:"), std::string::npos);
}

/* Quiet mode produces completely empty stdout. */
TEST_F(PredictTest, PredictQuietSuppressesAll) {
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(result.stdout_output.empty());
}

/* -o writes predictions, metrics, and proportions to JSON. */
TEST_F(PredictTest, PredictOutputFile) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_TRUE(j["metrics"].contains("error_rate"));
  EXPECT_TRUE(j["metrics"].contains("confusion_matrix"));
  EXPECT_TRUE(j.contains("proportions"));
}

/* Writing to an existing output file must fail. */
TEST_F(PredictTest, PredictOutputCollisionFails) {
  TempFile output;
  // Don't clear - file exists, should fail
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Forest output includes proportions by default. */
TEST_F(PredictTest, PredictForestIncludesProportions) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_TRUE(j.contains("proportions"));

  // 150 observations, each row is a vector of proportions
  auto& props = j["proportions"];
  EXPECT_EQ(props.size(), 150u);

  // Each row has one entry per group (iris has 3 groups)
  EXPECT_EQ(props[0].size(), 3u);

  // Each row sums to 1.0
  for (auto const& row : props) {
    double sum = 0;
    for (auto const& val : row) {
      sum += val.get<double>();
    }

    EXPECT_NEAR(sum, 1.0, 1e-6);
  }
}

/* --no-proportions omits proportions from forest output. */
TEST_F(PredictTest, PredictNoProportionsFlag) {
  TempFile output;
  output.clear();
  auto result =
      run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-proportions -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_FALSE(j.contains("proportions"));
}

// ---------------------------------------------------------------------------
// Predict subcommand — standalone tests
// ---------------------------------------------------------------------------

/* Single-tree output includes one-hot proportions. */
TEST(CLIPredict, PredictTreeModelIncludesProportions) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_TRUE(j.contains("proportions"));

  // 150 observations, each row is one-hot
  auto& props = j["proportions"];
  EXPECT_EQ(props.size(), 150u);
  EXPECT_EQ(props[0].size(), 3u);

  // Each row sums to 1.0 (one-hot for trees)
  for (auto const& row : props) {
    double sum = 0;
    for (auto const& val : row) {
      sum += val.get<double>();
    }

    EXPECT_NEAR(sum, 1.0, 1e-6);
  }
}

/* --no-proportions omits proportions from tree output. */
TEST(CLIPredict, PredictTreeNoProportionsFlag) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result =
      run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " --no-proportions -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_FALSE(j.contains("proportions"));
}

/* The nested confusion_matrix must not carry its own "error" key — error_rate
 * lives one level up under `metrics`. */
TEST(CLIPredict, PredictOutputNoErrorInConfusionMatrix) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  ASSERT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  ASSERT_TRUE(j["metrics"].contains("confusion_matrix"));
  EXPECT_FALSE(j["metrics"]["confusion_matrix"].contains("error"));
}

// ---------------------------------------------------------------------------
// Predict subcommand — label mapping and row order
// ---------------------------------------------------------------------------

namespace {
  void write_file(std::string const& path, std::string const& content) {
    std::ofstream out(path);
    out << content;
  }

  // Two perfectly separable classes: f1 < 10 → "a", f1 > 10 → "b".
  std::string const SEPARABLE_TRAIN_CSV = "f1,f2,y\n"
                                          "1,1,a\n2,1,a\n3,2,a\n"
                                          "11,5,b\n12,6,b\n13,5,b\n";
}

/* Labels are mapped through the model's group space: a prediction file
 * listing classes in a different order than the training file still yields
 * correct metrics. */
TEST(CLIPredictLabels, ReorderedClassesYieldCorrectMetrics) {
  TempFile train_csv(".csv");
  write_file(train_csv.path(), SEPARABLE_TRAIN_CSV);

  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  // Same rows, class "b" listed first.
  TempFile predict_csv(".csv");
  write_file(
      predict_csv.path(),
      "f1,f2,y\n"
      "11,5,b\n12,6,b\n13,5,b\n"
      "1,1,a\n2,1,a\n3,2,a\n"
  );

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_DOUBLE_EQ(j["metrics"]["error_rate"].get<double>(), 0.0);
}

/* predictions[i] corresponds to input row i — rows are not re-sorted. */
TEST(CLIPredictLabels, PredictionsPreserveInputRowOrder) {
  TempFile train_csv(".csv");
  write_file(train_csv.path(), SEPARABLE_TRAIN_CSV);

  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,f2,y\n11,5,b\n1,1,a\n12,6,b\n2,1,a\n");

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  ASSERT_EQ(j["predictions"].size(), 4U);
  EXPECT_EQ(j["predictions"][0], "b");
  EXPECT_EQ(j["predictions"][1], "a");
  EXPECT_EQ(j["predictions"][2], "b");
  EXPECT_EQ(j["predictions"][3], "a");
}

/* A label that was not in the training data fails with a clear error. */
TEST(CLIPredictLabels, UnknownLabelFails) {
  TempFile train_csv(".csv");
  write_file(train_csv.path(), SEPARABLE_TRAIN_CSV);

  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,f2,y\n1,1,c\n2,1,a\n");

  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path());
  EXPECT_NE(result.exit_code, 0);
  EXPECT_NE(result.stderr_output.find("training labels"), std::string::npos);
}

/* A class the data never carries has no error rate to report, so its
 * confusion-matrix row shows "-" rather than a division of zero by zero. */
TEST(CLIPredictLabels, ClassWithNoObservationsShowsDash) {
  TempFile train_csv(".csv");
  write_file(train_csv.path(), SEPARABLE_TRAIN_CSV);

  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  // Every row is labelled "a" but sits in "b" territory, so "b" is predicted
  // without ever appearing as an actual class.
  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,f2,y\n11,5,a\n12,6,a\n13,5,a\n");

  auto result = run_ppforest2("--no-color predict -M " + model.path() + " -d " + predict_csv.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  // The "b" row is all zeros and must not render as a nan percentage.
  EXPECT_EQ(result.stdout_output.find("nan"), std::string::npos) << result.stdout_output;
  EXPECT_NE(result.stdout_output.find("  -"), std::string::npos) << result.stdout_output;
}

/* Regression models parse the response as numeric even when every value is
 * integer-written — the model's mode, not the y column's written form,
 * decides the parse. */
TEST(CLIPredictLabels, RegressionIntegerResponsePredicts) {
  TempFile train_csv(".csv");
  write_file(train_csv.path(), "f1,f2,y\n1,1,10\n2,2,20\n3,3,30\n4,4,40\n5,5,50\n6,6,60\n");

  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train --mode regression -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + train_csv.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  ASSERT_EQ(j["predictions"].size(), 6U);
  EXPECT_TRUE(j["predictions"][0].is_number());
  ASSERT_TRUE(j.contains("metrics"));
}

// ---------------------------------------------------------------------------
// Predict subcommand — data with and without a response column
//
// Columns are matched to the model's features as `serve` matches a request:
// by name, or by position for a model saved without feature names. A column
// that is not a feature is the response; data without one has no metrics.
// ---------------------------------------------------------------------------

namespace {
  // A single tree trained on SEPARABLE_TRAIN_CSV (features f1, f2), saved
  // to @p model. Returns the exit code of `train`.
  int train_separable(TempFile& model) {
    TempFile train_csv(".csv");
    write_file(train_csv.path(), SEPARABLE_TRAIN_CSV);

    model.clear();
    return run_ppforest2("-q train -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path()).exit_code;
  }

  // Empties `meta.feature_names` in a saved model, as it is for a model
  // trained on data without column names.
  void strip_feature_names(TempFile const& model) {
    auto j                     = json::parse(model.read());
    j["meta"]["feature_names"] = json::array();
    write_file(model.path(), j.dump());
  }
}

/* The reported case: a crab model and a CSV with only its five features.
 * The last feature used to be read as the response ("Data has 4 feature
 * column(s) but the model was trained with 5"); now the rows are predicted,
 * without metrics. */
TEST(CLIPredictColumns, FeaturesOnlyPredictsWithoutMetrics) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + CRAB_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile predict_csv(".csv");
  write_file(
      predict_csv.path(),
      "FL,RW,CL,CW,BD\n"
      "8.1,6.7,16.1,19,7\n9,8.5,19.3,22.7,7.7\n9.1,6.9,16.7,18.6,7.4\n12.6,11.5,25,28.1,11.5\n"
  );

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(output.read());
  EXPECT_EQ(j["predictions"].size(), 4U);
  EXPECT_EQ(j["proportions"].size(), 4U);
  EXPECT_FALSE(j.contains("metrics"));
}

/* Without --output, data without a response still reports what was
 * predicted and why there are no metrics. */
TEST(CLIPredictColumns, FeaturesOnlyReportsPredictionCount) {
  TempFile model;
  ASSERT_EQ(train_separable(model), 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,f2\n1,1\n12,6\n");

  auto result = run_ppforest2("--no-color predict -M " + model.path() + " -d " + predict_csv.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;
  EXPECT_NE(result.stdout_output.find("Predicted 2 observations"), std::string::npos) << result.stdout_output;
  EXPECT_NE(result.stdout_output.find("no response column"), std::string::npos) << result.stdout_output;
  EXPECT_EQ(result.stdout_output.find("Error:"), std::string::npos) << result.stdout_output;
  EXPECT_NE(result.stdout_output.find("--output"), std::string::npos) << result.stdout_output;
}

/* One column more than the model's features: it is the response, and the
 * metrics are reported. */
TEST(CLIPredictColumns, ResponseColumnReportsMetrics) {
  TempFile model;
  ASSERT_EQ(train_separable(model), 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,f2,y\n1,1,a\n12,6,b\n");

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(output.read());
  EXPECT_EQ(j["predictions"], json::array({"a", "b"}));
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_DOUBLE_EQ(j["metrics"]["error_rate"].get<double>(), 0.0);
}

/* Columns are matched by name: features in another order predict the same,
 * and the response is found wherever it sits. */
TEST(CLIPredictColumns, ColumnsMatchedByName) {
  TempFile model;
  ASSERT_EQ(train_separable(model), 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "y,f2,f1\na,1,1\nb,6,12\n");

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(output.read());
  EXPECT_EQ(j["predictions"], json::array({"a", "b"}));
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_DOUBLE_EQ(j["metrics"]["error_rate"].get<double>(), 0.0);
}

/* Data lacking one of the model's features fails, naming it — here the
 * column count matches the model's, but the second column is the response. */
TEST(CLIPredictColumns, MissingFeatureFails) {
  TempFile model;
  ASSERT_EQ(train_separable(model), 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,y\n1,a\n12,b\n");

  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path());
  EXPECT_NE(result.exit_code, 0);
  EXPECT_NE(result.stderr_output.find("missing [f2]"), std::string::npos) << result.stderr_output;
}

/* A model saved without feature names takes the columns by position: its
 * features alone are predicted without metrics, and one column more is the
 * response. */
TEST(CLIPredictColumns, UnnamedModelTakesColumnsByPosition) {
  TempFile model;
  ASSERT_EQ(train_separable(model), 0);
  strip_feature_names(model);

  TempFile features_only(".csv");
  write_file(features_only.path(), "p,q\n1,1\n12,6\n");

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + features_only.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(output.read());
  EXPECT_EQ(j["predictions"], json::array({"a", "b"}));
  EXPECT_FALSE(j.contains("metrics"));

  TempFile with_response(".csv");
  write_file(with_response.path(), "p,q,r\n1,1,a\n12,6,b\n");

  TempFile labeled_output;
  labeled_output.clear();
  result =
      run_ppforest2("-q predict -M " + model.path() + " -d " + with_response.path() + " -o " + labeled_output.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  j = json::parse(labeled_output.read());
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_DOUBLE_EQ(j["metrics"]["error_rate"].get<double>(), 0.0);
}

/* Without feature names, a column count other than the model's feature
 * count or one more fails, stating the count expected. */
TEST(CLIPredictColumns, UnnamedModelWrongColumnCountFails) {
  TempFile model;
  ASSERT_EQ(train_separable(model), 0);
  strip_feature_names(model);

  for (std::string const csv : {"p\n1\n12\n", "p,q,r,s\n1,1,a,x\n12,6,b,y\n"}) {
    TempFile predict_csv(".csv");
    write_file(predict_csv.path(), csv);

    auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path());
    EXPECT_NE(result.exit_code, 0) << csv;
    EXPECT_NE(result.stderr_output.find("2 feature column(s)"), std::string::npos) << result.stderr_output;
  }
}

/* The last column that is not a feature is read as the response, so an
 * extra column that isn't one (an ID, say) fails, named; --no-metrics
 * doesn't read the response column. */
TEST(CLIPredictColumns, NoMetricsSkipsResponseColumn) {
  TempFile model;
  ASSERT_EQ(train_separable(model), 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "id,f1,f2\n1,1,1\n2,12,6\n");

  auto failed = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path());
  EXPECT_NE(failed.exit_code, 0);
  EXPECT_NE(failed.stderr_output.find("response column 'id'"), std::string::npos) << failed.stderr_output;
  EXPECT_NE(failed.stderr_output.find("Use --no-metrics"), std::string::npos) << failed.stderr_output;

  TempFile output;
  output.clear();
  auto result = run_ppforest2(
      "-q predict -M " + model.path() + " -d " + predict_csv.path() + " --no-metrics -o " + output.path()
  );
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(output.read());
  EXPECT_EQ(j["predictions"], json::array({"a", "b"}));
  EXPECT_FALSE(j.contains("metrics"));
}

/* A regression model predicts on its features alone, without metrics. */
TEST(CLIPredictColumns, RegressionFeaturesOnlyPredictsWithoutMetrics) {
  TempFile train_csv(".csv");
  write_file(train_csv.path(), "f1,f2,y\n1,1,10\n2,2,20\n3,3,30\n4,4,40\n5,5,50\n6,6,60\n");

  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train --mode regression -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,f2\n1,1\n6,6\n");

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path() + " -o " + output.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(output.read());
  ASSERT_EQ(j["predictions"].size(), 2U);
  EXPECT_TRUE(j["predictions"][0].is_number());
  EXPECT_FALSE(j.contains("metrics"));
  EXPECT_FALSE(j.contains("proportions"));
}

/* A regression response with a missing value is kept as text until metrics
 * need it: it fails the metrics, naming the row, and --no-metrics predicts
 * without reading it. */
TEST(CLIPredictColumns, RegressionResponseWithMissingValue) {
  TempFile train_csv(".csv");
  write_file(train_csv.path(), "f1,f2,y\n1,1,10\n2,2,20\n3,3,30\n4,4,40\n5,5,50\n6,6,60\n");

  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train --mode regression -d " + train_csv.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile predict_csv(".csv");
  write_file(predict_csv.path(), "f1,f2,y\n1,1,10\n6,6,NA\n");

  auto failed = run_ppforest2("-q predict -M " + model.path() + " -d " + predict_csv.path());
  EXPECT_NE(failed.exit_code, 0);
  EXPECT_NE(failed.stderr_output.find("Value 'NA' in response column 'y' (row 2)"), std::string::npos)
      << failed.stderr_output;
  EXPECT_NE(failed.stderr_output.find("Use --no-metrics"), std::string::npos) << failed.stderr_output;

  TempFile output;
  output.clear();
  auto result = run_ppforest2(
      "-q predict -M " + model.path() + " -d " + predict_csv.path() + " --no-metrics -o " + output.path()
  );
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(output.read());
  EXPECT_EQ(j["predictions"].size(), 2U);
  EXPECT_FALSE(j.contains("metrics"));
}

// ---------------------------------------------------------------------------
// Predict subcommand — error cases
// ---------------------------------------------------------------------------

/* Predict without -M must fail. */
TEST(CLIPredict, PredictMissingModelArgFails) {
  auto result = run_ppforest2("predict -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

/* Predict with a nonexistent model file must fail. */
TEST(CLIPredict, PredictNonexistentModelFails) {
  auto result = run_ppforest2("predict -M /nonexistent.json -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

/* Predicting with a single-tree model produces correct output. */
TEST(CLIPredict, PredictWithSingleTreeModel) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_EQ(j["predictions"].size(), 150u);
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_TRUE(j["metrics"].contains("error_rate"));
  EXPECT_TRUE(j["metrics"].contains("confusion_matrix"));
}
