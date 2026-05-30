#include "core/statistics.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace membench {

Statistics calculateStatistics(const std::vector<double>& values) {
    if (values.empty()) {
        return {};
    }

    Statistics stats;
    stats.average = std::accumulate(values.begin(), values.end(), 0.0) /
                    static_cast<double>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t midpoint = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        stats.median = (sorted[midpoint - 1] + sorted[midpoint]) / 2.0;
    } else {
        stats.median = sorted[midpoint];
    }
    stats.minimum = sorted.front();
    stats.maximum = sorted.back();

    double squared_sum = 0.0;
    for (double value : values) {
        const double delta = value - stats.average;
        squared_sum += delta * delta;
    }
    stats.stdev = std::sqrt(squared_sum / static_cast<double>(values.size()));
    return stats;
}

Statistics scaleStatistics(const Statistics& stats, double factor) {
    Statistics scaled = stats;
    scaled.average *= factor;
    scaled.median *= factor;
    scaled.minimum *= factor;
    scaled.maximum *= factor;
    scaled.stdev *= factor;
    return scaled;
}

}  // namespace membench
