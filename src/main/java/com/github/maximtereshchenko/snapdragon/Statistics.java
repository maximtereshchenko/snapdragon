package com.github.maximtereshchenko.snapdragon;

import java.util.List;

record Statistics(
    List<EpochTrainingStatistics> epochTrainingStatistics,
    List<EpochValidationStatistics> epochValidationStatistics
) {}
