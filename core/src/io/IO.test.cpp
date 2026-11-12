/**
 * @file IO.test.cpp
 * @brief Unit tests for CSV reading and file helper utilities.
 */
#include <gtest/gtest.h>

#include "io/IO.hpp"
#include "io/TempFile.hpp"
#include "utils/UserError.hpp"

#include <fstream>

using namespace ppforest2;

#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

static const std::string DATA_DIR  = PPFOREST2_DATA_DIR;
static std::string const IRIS_PATH = DATA_DIR + "/classification/iris.csv";

namespace {
  void write_csv(std::string const& path, std::string const& content) {
    std::ofstream out(path);
    out << content;
    out.close();
  }
}

TEST(CSVReadTest, AllNumericFeatures) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,label\n1.0,2.0,x\n3.0,4.0,y\n5.0,6.0,x\n");

  auto data = io::csv::read(tmp.path());

  EXPECT_EQ(data.x.rows(), 3);
  EXPECT_EQ(data.x.cols(), 2);
  EXPECT_FLOAT_EQ(data.x(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 4.0F);
  EXPECT_EQ(data.y(0), 0);
  EXPECT_EQ(data.y(1), 1);
  EXPECT_EQ(data.y(2), 0);
}

TEST(CSVReadTest, CategoricalFeatureColumn) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "color,size,label\nred,1.0,A\nblue,2.0,B\nred,3.0,A\ngreen,4.0,B\n");

  auto data = io::csv::read(tmp.path());

  EXPECT_EQ(data.x.rows(), 4);
  EXPECT_EQ(data.x.cols(), 2);

  // "color" column: red=0, blue=1, green=2 (order of first appearance)
  EXPECT_FLOAT_EQ(data.x(0, 0), 0.0F); // red
  EXPECT_FLOAT_EQ(data.x(1, 0), 1.0F); // blue
  EXPECT_FLOAT_EQ(data.x(2, 0), 0.0F); // red
  EXPECT_FLOAT_EQ(data.x(3, 0), 2.0F); // green

  // "size" column: numeric, unchanged
  EXPECT_FLOAT_EQ(data.x(0, 1), 1.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 2.0F);
}

TEST(CSVReadTest, MultipleCategoricalColumns) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "color,shape,val,label\nred,circle,1.0,X\nblue,square,2.0,Y\nred,circle,3.0,X\n");

  auto data = io::csv::read(tmp.path());

  EXPECT_EQ(data.x.rows(), 3);
  EXPECT_EQ(data.x.cols(), 3);

  // color: red=0, blue=1
  EXPECT_FLOAT_EQ(data.x(0, 0), 0.0F);
  EXPECT_FLOAT_EQ(data.x(1, 0), 1.0F);
  EXPECT_FLOAT_EQ(data.x(2, 0), 0.0F);

  // shape: circle=0, square=1
  EXPECT_FLOAT_EQ(data.x(0, 1), 0.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 1.0F);
  EXPECT_FLOAT_EQ(data.x(2, 1), 0.0F);

  // val: numeric
  EXPECT_FLOAT_EQ(data.x(0, 2), 1.0F);
}

TEST(CSVReadTest, GroupNamesPopulated) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "x,label\n1.0,setosa\n2.0,virginica\n3.0,setosa\n");

  auto data = io::csv::read(tmp.path());

  ASSERT_EQ(data.group_names.size(), 2U);
  EXPECT_EQ(data.group_names[0], "setosa");
  EXPECT_EQ(data.group_names[1], "virginica");
}

TEST(CSVReadTest, GroupNamesPreservedAfterSort) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "x,label\n1.0,B\n2.0,A\n3.0,B\n4.0,A\n");

  auto data = io::csv::read_sorted(tmp.path());

  ASSERT_EQ(data.group_names.size(), 2U);
  EXPECT_EQ(data.group_names[0], "B");
  EXPECT_EQ(data.group_names[1], "A");
}

// Mode auto-detection: regression iff any y value has fractional or
// scientific notation; integer-coded labels remain classification.

TEST(CSVReadTest, RegressionDetectedFromFractionalY) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,0.5\n3.0,4.0,1.7\n5.0,6.0,2.3\n");

  auto data = io::csv::read_sorted(tmp.path());

  EXPECT_TRUE(data.group_names.empty()) << "Fractional y should auto-detect as regression";
  // Sorted ascending by y → 0.5, 1.7, 2.3.
  EXPECT_FLOAT_EQ(data.y(0), 0.5F);
  EXPECT_FLOAT_EQ(data.y(1), 1.7F);
  EXPECT_FLOAT_EQ(data.y(2), 2.3F);
}

TEST(CSVReadTest, IntegerCodedLabelsRemainClassification) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,label\n1.0,2.0,1\n3.0,4.0,2\n5.0,6.0,3\n7.0,8.0,1\n");

  auto data = io::csv::read_sorted(tmp.path());

  ASSERT_EQ(data.group_names.size(), 3U) << "Integer-coded labels should map to classification group_names";
  EXPECT_EQ(data.group_names[0], "1");
  EXPECT_EQ(data.group_names[1], "2");
  EXPECT_EQ(data.group_names[2], "3");
}

TEST(CSVReadTest, ScientificNotationDetectsRegression) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,1e2\n3.0,4.0,2e2\n");

  auto data = io::csv::read_sorted(tmp.path());

  EXPECT_TRUE(data.group_names.empty()) << "y in scientific notation should auto-detect as regression";
  EXPECT_FLOAT_EQ(data.y(0), 100.0F);
  EXPECT_FLOAT_EQ(data.y(1), 200.0F);
}

TEST(CSVReadTest, NonNumericLabelStaysClassification) {
  // Already covered by AllNumericFeatures and GroupNamesPopulated, but
  // assert it explicitly: non-numeric y is unambiguously classification.
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,label\n1.0,2.0,foo\n3.0,4.0,bar\n");

  auto data = io::csv::read_sorted(tmp.path());

  ASSERT_EQ(data.group_names.size(), 2U);
}

TEST(CSVReadTest, CrabsDatasetWithCategoricalSex) {
  auto data = io::csv::read(PPFOREST2_DATA_DIR "/classification/crabs.csv");

  EXPECT_EQ(data.x.cols(), 7); // sex + index + 5 morphometrics
  EXPECT_EQ(data.x.rows(), 200);

  // "sex" is the first column and is categorical (M/F)
  // All values should be 0 or 1
  for (int i = 0; i < data.x.rows(); ++i) {
    types::Feature val = data.x(i, 0);
    EXPECT_TRUE(val == 0.0F || val == 1.0F) << "Row " << i << " sex=" << val;
  }
}

// ---------------------------------------------------------------------------
// File helpers — io::json::ensure_extension, check_*_not_exists
// ---------------------------------------------------------------------------

/**
 * @brief Death-test predicate: matches any non-zero exit code.
 */
class ExitedWithNonZero {
public:
  bool operator()(int exit_status) const {
    // clang-format off
    #ifdef _WIN32
    return exit_status != 0;
    #else
    return testing::ExitedWithCode(0)(exit_status) && WIFEXITED(exit_status);
    #endif
    // clang-format on
  }
};

/* Path already ending in .json is returned unchanged. */
TEST(CSVReadTest, RaggedRowInFileThrows) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,x\n3.0,y\n");

  EXPECT_THROW(io::csv::read(tmp.path()), UserError);
}

TEST(CSVReadTest, MissingValueInNumericColumnThrows) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,x\nNA,3.0,y\n");

  try {
    io::csv::read(tmp.path());
    FAIL() << "expected UserError";
  } catch (UserError const& e) {
    EXPECT_NE(std::string(e.what()).find("Column 'a'"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("'NA'"), std::string::npos) << e.what();
  }
}

TEST(CSVReadTest, EmptyCellInNumericColumnThrows) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,x\n,3.0,y\n");

  EXPECT_THROW(io::csv::read(tmp.path()), UserError);
}

TEST(CSVReadTest, NonFiniteCellsThrow) {
  io::TempFile nan_file(".csv");
  write_csv(nan_file.path(), "a,y\nnan,x\n2.0,y\n");
  EXPECT_THROW(io::csv::read(nan_file.path()), UserError);

  io::TempFile inf_file(".csv");
  write_csv(inf_file.path(), "a,y\ninf,x\n2.0,y\n");
  EXPECT_THROW(io::csv::read(inf_file.path()), UserError);
}

TEST(CSVReadTest, ForcedRegressionRejectsNanY) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,y\n1.0,nan\n2.0,2.5\n");

  EXPECT_THROW(io::csv::read(tmp.path(), types::Mode::Regression), UserError);
}

TEST(CSVReadTest, ForcedRegressionParsesIntegerY) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,10\n3.0,4.0,30\n5.0,6.0,20\n");

  auto data = io::csv::read(tmp.path(), types::Mode::Regression);

  EXPECT_TRUE(data.group_names.empty());
  EXPECT_FLOAT_EQ(data.y(0), 10.0F);
  EXPECT_FLOAT_EQ(data.y(1), 30.0F);
  EXPECT_FLOAT_EQ(data.y(2), 20.0F); // rows keep file order
}

TEST(CSVReadTest, ForcedRegressionRejectsNonNumericY) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,x\n3.0,4.0,y\n");

  EXPECT_THROW(io::csv::read(tmp.path(), types::Mode::Regression), UserError);
}

TEST(CSVReadTest, ForcedClassificationKeepsRowOrder) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,B\n3.0,4.0,A\n5.0,6.0,B\n");

  auto data = io::csv::read(tmp.path(), types::Mode::Classification);

  ASSERT_EQ(data.group_names.size(), 2U);
  EXPECT_EQ(data.group_names[0], "B"); // first appearance
  EXPECT_EQ(data.y(0), 0);
  EXPECT_EQ(data.y(1), 1);
  EXPECT_EQ(data.y(2), 0);
}

TEST(RemapLabels, RemapsIntoTargetOrder) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,B\n3.0,4.0,A\n5.0,6.0,B\n");

  auto data = io::csv::read(tmp.path(), types::Mode::Classification);
  auto y    = io::csv::remap_labels(data, {"A", "B"});

  EXPECT_EQ(y(0), 1); // B → 1 in the target space
  EXPECT_EQ(y(1), 0); // A → 0
  EXPECT_EQ(y(2), 1);
}

TEST(RemapLabels, UnknownLabelThrows) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,C\n3.0,4.0,A\n");

  auto data = io::csv::read(tmp.path(), types::Mode::Classification);

  EXPECT_THROW(io::csv::remap_labels(data, {"A", "B"}), UserError);
}

TEST(FileHelpers, EnsureJsonExtensionWithExtension) {
  EXPECT_EQ(io::json::ensure_extension("model.json"), "model.json");
}

/* Path without extension gets .json appended. */
TEST(FileHelpers, EnsureJsonExtensionWithoutExtension) {
  EXPECT_EQ(io::json::ensure_extension("model"), "model.json");
}

/* Non-.json extension gets .json added (e.g. .txt -> .txt.json). */
TEST(FileHelpers, EnsureJsonExtensionWithOtherExtension) {
  EXPECT_EQ(io::json::ensure_extension("model.txt"), "model.txt.json");
}

/* Full path without extension gets .json appended. */
TEST(FileHelpers, EnsureJsonExtensionWithPath) {
  EXPECT_EQ(io::json::ensure_extension("/tmp/model"), "/tmp/model.json");
}

/* check_file_not_exists succeeds for a nonexistent path. */
TEST(FileHelpers, CheckFileNotExistsOnNonexistent) {
  io::check_file_not_exists("/nonexistent/path/that/doesnt/exist.json");
}

/* check_file_not_exists throws for an existing file. */
TEST(FileHelpers, CheckFileNotExistsOnExisting) {
  EXPECT_THROW(io::check_file_not_exists(IRIS_PATH), ppforest2::UserError);
}

/* check_dir_not_exists succeeds for a nonexistent path. */
TEST(FileHelpers, CheckDirNotExistsOnNonexistent) {
  io::check_dir_not_exists("/nonexistent/path/that/doesnt/exist");
}

/* check_dir_not_exists throws for an existing directory. */
TEST(FileHelpers, CheckDirNotExistsOnExisting) {
  EXPECT_THROW(io::check_dir_not_exists(DATA_DIR), ppforest2::UserError);
}

/* check_file_exists succeeds for an existing file. */
TEST(FileHelpers, CheckFileExistsOnExisting) {
  io::check_file_exists(IRIS_PATH);
}

/* check_file_exists throws for a nonexistent file. */
TEST(FileHelpers, CheckFileExistsOnNonexistent) {
  EXPECT_THROW(io::check_file_exists("/nonexistent/path.csv"), ppforest2::UserError);
}

// ---------------------------------------------------------------------------
// read_prediction_data — columns matched to a model's features
// ---------------------------------------------------------------------------

namespace {
  types::Names const FEATURES = {"a", "b", "c"};

  // The message of the UserError @p f throws, or "" if it throws none.
  template<typename F> std::string user_error_message(F&& f) {
    try {
      f();
    } catch (UserError const& e) {
      return e.what();
    }
    return "";
  }
}

/* Only the model's features: no response, and x follows the model's column
 * order whatever the order in the file. */
TEST(ReadPredictionData, MatchesFeaturesByName) {
  auto const data = io::csv::read_prediction_data_from_string("c,a,b\n3,1,2\n6,4,5\n", FEATURES, 3);

  ASSERT_EQ(data.x.rows(), 2);
  ASSERT_EQ(data.x.cols(), 3);
  EXPECT_FLOAT_EQ(data.x(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(data.x(0, 1), 2.0F);
  EXPECT_FLOAT_EQ(data.x(0, 2), 3.0F);
  EXPECT_FLOAT_EQ(data.x(1, 2), 6.0F);
  EXPECT_FALSE(data.response.has_value());
}

/* One column more than the features: it is the response, kept as text. */
TEST(ReadPredictionData, ExtraColumnIsTheResponse) {
  auto const data = io::csv::read_prediction_data_from_string("a,b,c,y\n1,2,3,x\n4,5,6,z\n", FEATURES, 3);

  ASSERT_EQ(data.x.cols(), 3);
  ASSERT_TRUE(data.response.has_value());
  EXPECT_EQ(data.response->name, "y");
  EXPECT_EQ(data.response->values, (std::vector<std::string>{"x", "z"}));
}

/* With several columns that are not features, the last one is the response
 * and the others are ignored. */
TEST(ReadPredictionData, LastNonFeatureColumnIsTheResponse) {
  auto const data =
      io::csv::read_prediction_data_from_string("id,b,a,notes,c,y\n7,2,1,low,3,x\n8,5,4,high,6,z\n", FEATURES, 3);

  EXPECT_FLOAT_EQ(data.x(1, 0), 4.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 5.0F);
  EXPECT_FLOAT_EQ(data.x(1, 2), 6.0F);
  ASSERT_TRUE(data.response.has_value());
  EXPECT_EQ(data.response->name, "y");
}

/* A feature column is converted with the same rules as training data: a mix
 * of numeric and non-numeric cells is an error. */
TEST(ReadPredictionData, MixedFeatureColumnThrows) {
  auto const message =
      user_error_message([] { io::csv::read_prediction_data_from_string("a,b,c\n1,2,3\nNA,5,6\n", FEATURES, 3); });
  EXPECT_NE(message.find("Column 'a'"), std::string::npos) << message;
}

/* A feature missing from the header is an error naming it. */
TEST(ReadPredictionData, MissingFeatureThrows) {
  auto const message =
      user_error_message([] { io::csv::read_prediction_data_from_string("a,c,y\n1,3,x\n", FEATURES, 3); });
  EXPECT_NE(message.find("missing [b]"), std::string::npos) << message;
}

/* Without feature names, as many columns as features are taken by position,
 * whatever the header says, and there is no response. */
TEST(ReadPredictionData, UnnamedModelTakesColumnsByPosition) {
  auto const data = io::csv::read_prediction_data_from_string("p,q\n1,2\n3,4\n", {}, 2);

  ASSERT_EQ(data.x.cols(), 2);
  EXPECT_FLOAT_EQ(data.x(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 4.0F);
  EXPECT_FALSE(data.response.has_value());
}

/* Without feature names, one column more than the features is the response. */
TEST(ReadPredictionData, UnnamedModelLastExtraColumnIsTheResponse) {
  auto const data = io::csv::read_prediction_data_from_string("p,q,r\n1,2,x\n3,4,z\n", {}, 2);

  ASSERT_EQ(data.x.cols(), 2);
  EXPECT_FLOAT_EQ(data.x(1, 1), 4.0F);
  ASSERT_TRUE(data.response.has_value());
  EXPECT_EQ(data.response->name, "r");
  EXPECT_EQ(data.response->values, (std::vector<std::string>{"x", "z"}));
}

/* Without feature names, any other column count is an error stating the
 * expected one. */
TEST(ReadPredictionData, UnnamedModelWrongColumnCountThrows) {
  auto const fewer = user_error_message([] { io::csv::read_prediction_data_from_string("p\n1\n", {}, 2); });
  EXPECT_NE(fewer.find("2 feature column(s)"), std::string::npos) << fewer;
  EXPECT_NE(fewer.find("got 1"), std::string::npos) << fewer;

  auto const more = user_error_message([] { io::csv::read_prediction_data_from_string("p,q,r,s\n1,2,3,4\n", {}, 2); });
  EXPECT_NE(more.find("got 4"), std::string::npos) << more;
}

/* The file reader matches columns the same way: iris.csv carries its
 * response, Type, after the features. */
TEST(ReadPredictionData, ReadsFile) {
  types::Names const iris_features = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};
  auto const data                  = io::csv::read_prediction_data(IRIS_PATH, iris_features, 4);

  EXPECT_EQ(data.x.rows(), 150);
  EXPECT_EQ(data.x.cols(), 4);
  EXPECT_FLOAT_EQ(data.x(0, 0), 5.1F);
  ASSERT_TRUE(data.response.has_value());
  EXPECT_EQ(data.response->name, "Type");
  EXPECT_EQ(data.response->values.front(), "setosa");
}

TEST(ReadPredictionData, EmptyBodyThrows) {
  EXPECT_THROW(io::csv::read_prediction_data_from_string("", FEATURES, 3), UserError);
}

TEST(ReadPredictionData, HeaderOnlyThrows) {
  EXPECT_THROW(io::csv::read_prediction_data_from_string("a,b,c\n", FEATURES, 3), UserError);
}

TEST(ReadPredictionData, RaggedRowsThrow) {
  EXPECT_THROW(io::csv::read_prediction_data_from_string("a,b,c\n1,2,3\n4,5\n", FEATURES, 3), UserError);
}

TEST(ReadPredictionData, EmptyFileThrows) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "");

  EXPECT_THROW(io::csv::read_prediction_data(tmp.path(), FEATURES, 3), UserError);
}

// ---------------------------------------------------------------------------
// parse_response — a response column in the model's outcome space
// ---------------------------------------------------------------------------

/* Labels become indices into the model's labels, whatever order the data
 * lists them in. */
TEST(ParseResponse, ClassificationUsesModelLabelOrder) {
  io::csv::ResponseColumn const response{"y", {"b", "a", "b"}};

  auto const y = io::csv::parse_response(response, types::Mode::Classification, {"a", "b"});

  ASSERT_EQ(y.size(), 3);
  EXPECT_EQ(y(0), 1);
  EXPECT_EQ(y(1), 0);
  EXPECT_EQ(y(2), 1);
}

/* A label the model wasn't trained on is an error naming it and its column. */
TEST(ParseResponse, UnknownLabelThrows) {
  io::csv::ResponseColumn const response{"y", {"a", "c"}};

  auto const message =
      user_error_message([&] { io::csv::parse_response(response, types::Mode::Classification, {"a", "b"}); });
  EXPECT_NE(message.find("'c'"), std::string::npos) << message;
  EXPECT_NE(message.find("column 'y'"), std::string::npos) << message;
  EXPECT_NE(message.find("row 2"), std::string::npos) << message;
}

TEST(ParseResponse, RegressionParsesNumbers) {
  io::csv::ResponseColumn const response{"y", {"1.5", "2", "-3e2"}};

  auto const y = io::csv::parse_response(response, types::Mode::Regression, {});

  ASSERT_EQ(y.size(), 3);
  EXPECT_FLOAT_EQ(y(0), 1.5F);
  EXPECT_FLOAT_EQ(y(1), 2.0F);
  EXPECT_FLOAT_EQ(y(2), -300.0F);
}

TEST(ParseResponse, RegressionNonNumericThrows) {
  io::csv::ResponseColumn const response{"y", {"1.5", "NA"}};

  auto const message = user_error_message([&] { io::csv::parse_response(response, types::Mode::Regression, {}); });
  EXPECT_NE(message.find("'NA'"), std::string::npos) << message;
  EXPECT_NE(message.find("not numeric"), std::string::npos) << message;
}

/* A number beyond float's range is a UserError, not a stray std::out_of_range
 * (`serve` only catches UserError to omit metrics). */
TEST(ParseResponse, RegressionOutOfRangeThrowsUserError) {
  io::csv::ResponseColumn const response{"y", {"1e300"}};

  EXPECT_THROW(io::csv::parse_response(response, types::Mode::Regression, {}), UserError);
}
