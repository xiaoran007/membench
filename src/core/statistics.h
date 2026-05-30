#ifndef MEMBENCH_CORE_STATISTICS_H
#define MEMBENCH_CORE_STATISTICS_H

#include "core/types.h"

#include <vector>

namespace membench {

Statistics calculateStatistics(const std::vector<double>& values);
Statistics scaleStatistics(const Statistics& stats, double factor);

}  // namespace membench

#endif  // MEMBENCH_CORE_STATISTICS_H
