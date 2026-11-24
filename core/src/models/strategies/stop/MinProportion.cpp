#include "models/strategies/stop/MinProportion.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/JsonReader.hpp"
#include "utils/UserError.hpp"

#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

using namespace ppforest2::types;

namespace ppforest2::stop {
  MinProportion::MinProportion(Feature proportion)
      : proportion(proportion) {
    user_error(proportion > 0 && proportion < 1, "stop_min_proportion: proportion must be in (0, 1)");
  }

  nlohmann::json MinProportion::to_json() const {
    return {{"name", "min_proportion"}, {"proportion", proportion}};
  }

  std::string MinProportion::display_name() const {
    std::ostringstream oss;
    oss << "Min proportion (" << std::defaultfloat << std::setprecision(6) << proportion << ")";
    return oss.str();
  }

  bool MinProportion::compute(NodeContext const& ctx, stats::RNG& /*rng*/) const {
    double const share = static_cast<double>(ctx.y.total_size()) / static_cast<double>(ctx.x.rows());

    return share <= static_cast<double>(proportion);
  }

  StopRule::Ptr min_proportion(Feature proportion) {
    return std::make_shared<MinProportion>(proportion);
  }

  StopRule::Ptr MinProportion::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "min_proportion"};
    r.only_keys({"name", "proportion"});
    return min_proportion(static_cast<Feature>(r.require_number("proportion", 0.0, 1.0)));
  }
}
