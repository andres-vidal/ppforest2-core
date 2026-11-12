/**
 * @file IO.cpp
 * @brief File I/O utilities, JSON and CSV reading/writing.
 */
#include "io/IO.hpp"
#include "stats/Stats.hpp"
#include "utils/UserError.hpp"

#include <csv.hpp> // IWYU pragma: keep

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::io {
  void check_file_exists(std::string const& path) {
    user_error(std::filesystem::exists(path), fmt::format("File not found: {}", path));
  }

  void check_file_not_exists(std::string const& path) {
    user_error(!std::filesystem::exists(path), fmt::format("File already exists: {}", path));
  }

  void check_dir_not_exists(std::string const& path) {
    user_error(!std::filesystem::exists(path), fmt::format("Directory already exists: {}", path));
  }
}

namespace ppforest2::io::json {
  std::string ensure_extension(std::string const& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
      return path;
    }

    return path + ".json";
  }

  nlohmann::json read_file(std::string const& path, ErrorHandler on_error) {
    std::ifstream in(path);

    on_error(in.is_open(), fmt::format("Could not open file: {}", path));

    try {
      return nlohmann::json::parse(in);
    } catch (nlohmann::json::parse_error const& e) {
      on_error(false, fmt::format("Invalid JSON in file: {}", e.what()));
      throw; // unreachable — on_error always throws
    }
  }

  void write_file(nlohmann::json const& data, std::string const& path, ErrorHandler on_error) {
    std::ofstream out(path);

    on_error(out.is_open(), fmt::format("Could not open file for writing: {}", path));

    out << data.dump(2);
    out.close();
  }
}

namespace ppforest2::io::text {
  void write_file(std::string const& content, std::string const& path, ErrorHandler on_error) {
    std::ofstream out(path);

    on_error(out.is_open(), fmt::format("Could not open file for writing: {}", path));

    out << content;
    out.close();
  }
}

namespace ppforest2::io::csv {
  namespace {
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Signed-int subscript into a `std::vector` (or any container with
    // `operator[]`). Tightens the per-callsite `static_cast<std::size_t>(j)`
    // noise into a single named conversion. Forwards through the proxy
    // reference returned by `std::vector<bool>` correctly.
    template<typename Container> decltype(auto) at(Container&& v, int j) {
      return std::forward<Container>(v)[static_cast<std::size_t>(j)];
    }

    // Note: non-finite values ("nan", "inf" — accepted by strtod) do not
    // count as numeric. A NaN feature poisons every projection it touches,
    // so such cells must surface as errors, not silently flow into training.
    bool is_numeric(std::string const& s) {
      if (s.empty()) {
        return false;
      }

      char* end      = nullptr;
      double const v = std::strtod(s.c_str(), &end);
      return end != s.c_str() && *end == '\0' && std::isfinite(v);
    }

    bool is_floating(std::string const& s) {
      return is_numeric(s) && s.find_first_of(".eE") != std::string::npos;
    }

    struct CategoricalEncoder {
      std::unordered_map<std::string, int> mapping;

      Feature encode(std::string const& value) {
        auto it = mapping.find(value);

        if (it == mapping.end()) {
          int code       = static_cast<int>(mapping.size());
          mapping[value] = code;
          return static_cast<Feature>(code);
        }

        return static_cast<Feature>(it->second);
      }
    };

    // -----------------------------------------------------------------------
    // Parse pipeline
    //
    // CSV → (raw string rows) → (column-type detection + mode detection) →
    // (typed FeatureMatrix / OutcomeVector + group_names). Mode is detected
    // from the y column's written form: any value with fractional /
    // scientific notation (`.`, `e`, `E`) routes to regression (y as float,
    // empty group_names); otherwise y maps to integer codes (classification,
    // with group_names populated). Each phase below is one helper.
    // -----------------------------------------------------------------------

    struct RawRows {
      std::vector<std::vector<std::string>> values;
      int n_cols = 0;
    };

    struct ParsedCSV {
      FeatureMatrix x;
      OutcomeVector y;
      types::Names group_names;
      types::Names feature_names;
    };

    RawRows read_raw_rows(::csv::CSVReader& reader, int expected_cols) {
      RawRows raw;
      raw.n_cols = expected_cols;

      for (::csv::CSVRow& row : reader) {
        int const row_num = static_cast<int>(raw.values.size()) + 1;

        user_error(
            static_cast<int>(row.size()) == raw.n_cols,
            fmt::format("Row {} has {} column(s), expected {} (same as the header)", row_num, row.size(), raw.n_cols)
        );

        std::vector<std::string> values;
        values.reserve(row.size());
        for (auto&& cell : row) {
          values.push_back(cell.get<std::string>());
        }
        raw.values.push_back(std::move(values));
      }

      return raw;
    }

    // A column is categorical only when EVERY cell fails `is_numeric`. A mix
    // of numeric and non-numeric cells is almost always a data problem (a
    // stray "NA", an empty cell, a "nan"/"inf") — silently integer-encoding
    // the whole column would destroy its numeric values, so reject it with
    // the first offending cell named.
    std::vector<bool> is_categorical(RawRows const& raw, int n_features, types::Names const& feature_names) {
      std::vector<bool> res(static_cast<std::size_t>(n_features), false);

      for (int j = 0; j < n_features; ++j) {
        bool any_numeric  = false;
        int offending_row = -1;
        std::string offending_cell;

        for (int i = 0; i < static_cast<int>(raw.values.size()); ++i) {
          auto const& cell = at(at(raw.values, i), j);
          if (is_numeric(cell)) {
            any_numeric = true;
          } else if (offending_row < 0) {
            offending_row  = i;
            offending_cell = cell;
          }
        }

        user_error(
            !(any_numeric && offending_row >= 0),
            fmt::format(
                "Column '{}' mixes numeric and non-numeric values (row {} has '{}'). Missing or non-finite cells "
                "are not supported — clean the data, or make the column fully categorical.",
                at(feature_names, j),
                offending_row + 1,
                offending_cell
            )
        );

        at(res, j) = offending_row >= 0;
      }

      return res;
    }

    // Auto-detect regression: any y value with fractional / scientific
    // notation marks y as continuous. Integer-coded labels (e.g. Wine's
    // "1", "2", "3") look numeric but stay on the classification path.
    bool is_classification(RawRows const& raw) {
      int const y_col = raw.n_cols - 1;
      return std::all_of(raw.values.begin(), raw.values.end(), [&](auto const& row) {
        return !is_floating(at(row, y_col));
      });
    }

    Outcome encode_classification_y(
        std::string const& y_str, types::Names& group_names, std::unordered_map<std::string, int>& label_mapping
    ) {
      auto [it, inserted] = label_mapping.try_emplace(y_str, static_cast<int>(group_names.size()));
      if (inserted) {
        group_names.push_back(y_str);
      }
      return static_cast<Outcome>(it->second);
    }

    Outcome parse_regression_y(std::string const& y_str, int row_num) {
      user_error(
          is_numeric(y_str), fmt::format("Row {} response value '{}' is not numeric (regression mode)", row_num, y_str)
      );
      return std::stof(y_str);
    }

    ParsedCSV encode_rows(RawRows const& raw, types::Names feature_names, std::optional<types::Mode> mode) {
      int const n_features = raw.n_cols - 1;
      int const y_col      = raw.n_cols - 1;
      int const n          = static_cast<int>(raw.values.size());

      auto const _is_classification = mode ? types::is_classification(*mode) : is_classification(raw);
      auto const _is_categorical    = is_categorical(raw, n_features, feature_names);

      std::vector<CategoricalEncoder> encoders(static_cast<std::size_t>(n_features));
      FeatureMatrix x(n, n_features);
      OutcomeVector y(n);
      types::Names group_names;
      std::unordered_map<std::string, int> label_mapping;

      for (int i = 0; i < n; ++i) {
        auto const& row = at(raw.values, i);

        for (int j = 0; j < n_features; ++j) {
          auto const& val = at(row, j);
          x(i, j)         = at(_is_categorical, j) ? at(encoders, j).encode(val) : std::stof(val);
        }

        auto const& y_str = at(row, y_col);

        if (_is_classification) {
          y(i) = encode_classification_y(y_str, group_names, label_mapping);
        } else {
          y(i) = parse_regression_y(y_str, i + 1);
        }
      }

      return {std::move(x), std::move(y), std::move(group_names), std::move(feature_names)};
    }

    // Without THROW, csv-parser's default policy silently drops rows whose
    // column count differs from the header — training would proceed on
    // partial data with no diagnostic.
    ::csv::CSVFormat strict_format() {
      ::csv::CSVFormat format;
      format.variable_columns(::csv::VariableColumnPolicy::THROW);
      return format;
    }

    ParsedCSV parse_csv(std::string const& filename, std::optional<types::Mode> mode = std::nullopt) {
      ::csv::CSVReader reader(filename, strict_format());

      auto const col_names = reader.get_col_names();
      user_error(col_names.size() >= 2, "CSV header must have at least 2 columns (features + response)");

      int const n_cols = static_cast<int>(col_names.size());
      types::Names feature_names(col_names.begin(), col_names.end() - 1);

      RawRows raw = read_raw_rows(reader, n_cols);
      user_error(!raw.values.empty(), "CSV file is empty or has no data rows");

      return encode_rows(raw, std::move(feature_names), mode);
    }

    // Encodes every column of a CSV to predict on, as training data is
    // encoded: numeric columns are parsed, categorical ones integer-encoded
    // in first-appearance order. `names` holds the header names, for error
    // messages; the caller picks the model's features out of the result.
    FeatureMatrix encode_features(RawRows const& raw, types::Names const& names) {
      int const n_cols = raw.n_cols;
      int const n      = static_cast<int>(raw.values.size());

      auto const _is_categorical = is_categorical(raw, n_cols, names);

      std::vector<CategoricalEncoder> encoders(static_cast<std::size_t>(n_cols));
      FeatureMatrix x(n, n_cols);

      for (int i = 0; i < n; ++i) {
        auto const& row = at(raw.values, i);

        for (int j = 0; j < n_cols; ++j) {
          auto const& val = at(row, j);
          x(i, j)         = at(_is_categorical, j) ? at(encoders, j).encode(val) : std::stof(val);
        }
      }

      return x;
    }

    // -----------------------------------------------------------------------
    // Prediction data
    //
    // A file to predict on is matched against the model by its header, which
    // decides which columns are the model's features and which one, if any,
    // is the response. Every column is encoded, as in training data, and the
    // features are then taken in the model's order; the response also keeps
    // its text until the caller parses it against the model's labels.
    // -----------------------------------------------------------------------

    // Which CSV columns hold the model's features, in the model's column
    // order, and which one, if any, holds the response.
    struct ColumnMatch {
      std::vector<int> features;
      std::optional<int> response;
    };

    // A model without feature names can only take the columns by position:
    // its features in training order, optionally followed by the response.
    ColumnMatch match_by_position(int n_cols, int n_features) {
      user_error(
          n_cols == n_features || n_cols == n_features + 1,
          fmt::format(
              "Feature mismatch: the model has no feature names, so the data must have its {} feature column(s) in "
              "training order, optionally followed by the response (got {} column(s))",
              n_features,
              n_cols
          )
      );

      ColumnMatch match;
      match.features.resize(static_cast<std::size_t>(n_features));
      std::iota(match.features.begin(), match.features.end(), 0);

      if (n_cols > n_features) {
        match.response = n_features;
      }

      return match;
    }

    // Features are matched by name, so their order in the file doesn't
    // matter. The response is the last column that isn't a feature, which is
    // where a training CSV has it; any other extra column (an ID, say) is
    // ignored.
    ColumnMatch match_by_name(types::Names const& header, types::Names const& feature_names) {
      std::unordered_map<std::string, int> column_of;
      for (int j = 0; j < static_cast<int>(header.size()); ++j) {
        column_of.emplace(at(header, j), j);
      }

      ColumnMatch match;
      types::Names missing;

      for (auto const& name : feature_names) {
        auto const it = column_of.find(name);
        if (it == column_of.end()) {
          missing.push_back(name);
        } else {
          match.features.push_back(it->second);
        }
      }

      user_error(
          missing.empty(),
          fmt::format(
              "Feature mismatch: missing [{}]. Columns are matched to the model's features by name.",
              fmt::join(missing, ", ")
          )
      );

      std::unordered_set<std::string> const feature_set(feature_names.begin(), feature_names.end());
      for (int j = static_cast<int>(header.size()) - 1; j >= 0; --j) {
        if (feature_set.count(at(header, j)) == 0) {
          match.response = j;
          break;
        }
      }

      return match;
    }

    // Column j of the result is column `columns[j]` of @p x.
    FeatureMatrix select_columns(FeatureMatrix const& x, std::vector<int> const& columns) {
      FeatureMatrix selected(x.rows(), static_cast<int>(columns.size()));
      for (int j = 0; j < static_cast<int>(columns.size()); ++j) {
        selected.col(j) = x.col(at(columns, j));
      }
      return selected;
    }

    PredictionData read_prediction_csv(::csv::CSVReader& reader, types::Names const& feature_names, int n_features) {
      auto const col_names = reader.get_col_names();
      user_error(!col_names.empty(), "CSV header must have at least 1 column");

      types::Names const header(col_names.begin(), col_names.end());
      int const n_cols = static_cast<int>(header.size());

      ColumnMatch const match =
          feature_names.empty() ? match_by_position(n_cols, n_features) : match_by_name(header, feature_names);

      RawRows const raw = read_raw_rows(reader, n_cols);
      user_error(!raw.values.empty(), "CSV has no data rows");

      PredictionData data;
      data.x = select_columns(encode_features(raw, header), match.features);

      if (match.response) {
        ResponseColumn response{at(header, *match.response), {}};
        response.values.reserve(raw.values.size());
        for (auto const& row : raw.values) {
          response.values.push_back(at(row, *match.response));
        }
        data.response = std::move(response);
      }

      return data;
    }

    // Picks the `DataPacket` ctor by mode: classification derives groups
    // from `y` codes (0..G-1); regression has no group concept and passes `{}`.
    DataPacket make_data_packet(ParsedCSV const& parsed) {
      if (parsed.group_names.empty()) {
        return DataPacket(parsed.x, parsed.y, {}, {}, parsed.feature_names);
      } else {
        return DataPacket(parsed.x, parsed.y, parsed.group_names, parsed.feature_names);
      }
    }
  }

  namespace {
    DataPacket read_impl(std::string const& filename, std::optional<types::Mode> mode) {
      try {
        return make_data_packet(parse_csv(filename, mode));
      } catch (UserError const&) {
        throw;
      } catch (std::exception const& e) {
        throw UserError(fmt::format("Error reading CSV file '{}': {}", filename, e.what()));
      }
    }
  }

  DataPacket read(std::string const& filename) {
    return read_impl(filename, std::nullopt);
  }

  DataPacket read(std::string const& filename, types::Mode mode) {
    return read_impl(filename, mode);
  }

  void check_feature_count(DataPacket const& data, int n_features) {
    user_error(
        static_cast<int>(data.x.cols()) == n_features,
        fmt::format("Data has {} feature column(s) but the model was trained with {}.", data.x.cols(), n_features)
    );
  }

  OutcomeVector remap_labels(DataPacket const& data, types::Names const& target_groups) {
    std::unordered_map<std::string, int> target_codes;
    for (int i = 0; i < static_cast<int>(target_groups.size()); ++i) {
      target_codes.emplace(at(target_groups, i), i);
    }

    OutcomeVector remapped(data.y.size());
    for (int i = 0; i < static_cast<int>(data.y.size()); ++i) {
      auto const& label = at(data.group_names, static_cast<int>(data.y(i)));
      auto const it     = target_codes.find(label);
      user_error(
          it != target_codes.end(),
          fmt::format("Label '{}' in the data is not one of the model's training labels.", label)
      );
      remapped(i) = static_cast<Outcome>(it->second);
    }

    return remapped;
  }

  namespace {
    DataPacket read_sorted_impl(std::string const& filename, std::optional<types::Mode> mode) {
      try {
        auto parsed = parse_csv(filename, mode);
        sort(parsed.x, parsed.y);
        return make_data_packet(parsed);
      } catch (UserError const&) {
        throw;
      } catch (std::exception const& e) {
        throw UserError(fmt::format("Error reading CSV file '{}': {}", filename, e.what()));
      }
    }
  }

  DataPacket read_sorted(std::string const& filename) {
    return read_sorted_impl(filename, std::nullopt);
  }

  DataPacket read_sorted(std::string const& filename, types::Mode mode) {
    return read_sorted_impl(filename, mode);
  }

  void write(DataPacket const& data, std::string const& filename) {
    std::ofstream out(filename);

    user_error(out.is_open(), fmt::format("Could not open file for writing: {}", filename));

    Eigen::IOFormat const csv_row(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",");

    for (Eigen::Index i = 0; i < data.x.rows(); ++i) {
      out << data.x.row(i).format(csv_row) << "," << data.y[i] << "\n";
    }
  }

  PredictionData read_prediction_data(std::string const& filename, types::Names const& feature_names, int n_features) {
    try {
      ::csv::CSVReader reader(filename, strict_format());
      return read_prediction_csv(reader, feature_names, n_features);
    } catch (UserError const&) {
      throw;
    } catch (std::exception const& e) {
      throw UserError(fmt::format("Error reading CSV file '{}': {}", filename, e.what()));
    }
  }

  PredictionData
  read_prediction_data_from_string(std::string const& content, types::Names const& feature_names, int n_features) {
    try {
      user_error(!content.empty(), "CSV body is empty");

      std::stringstream stream(content);
      ::csv::CSVReader reader(stream, strict_format());
      return read_prediction_csv(reader, feature_names, n_features);
    } catch (UserError const&) {
      throw;
    } catch (std::exception const& e) {
      throw UserError(fmt::format("Error parsing CSV: {}", e.what()));
    }
  }

  OutcomeVector parse_response(ResponseColumn const& response, types::Mode mode, types::Names const& groups) {
    try {
      int const n = static_cast<int>(response.values.size());
      OutcomeVector y(n);

      if (types::is_classification(mode)) {
        std::unordered_map<std::string, int> code_of;
        for (int k = 0; k < static_cast<int>(groups.size()); ++k) {
          code_of.emplace(at(groups, k), k);
        }

        for (int i = 0; i < n; ++i) {
          auto const& label = at(response.values, i);
          auto const it     = code_of.find(label);
          user_error(
              it != code_of.end(),
              fmt::format(
                  "Label '{}' in response column '{}' (row {}) is not one of the model's training labels.",
                  label,
                  response.name,
                  i + 1
              )
          );
          y(i) = static_cast<Outcome>(it->second);
        }
      } else {
        for (int i = 0; i < n; ++i) {
          auto const& value = at(response.values, i);
          user_error(
              is_numeric(value),
              fmt::format("Value '{}' in response column '{}' (row {}) is not numeric.", value, response.name, i + 1)
          );
          y(i) = std::stof(value);
        }
      }

      return y;
    } catch (UserError const&) {
      throw;
    } catch (std::exception const& e) {
      // std::stof rejects magnitudes beyond float's range that `is_numeric` accepts.
      throw UserError(fmt::format("Error parsing response column '{}': {}", response.name, e.what()));
    }
  }
}
