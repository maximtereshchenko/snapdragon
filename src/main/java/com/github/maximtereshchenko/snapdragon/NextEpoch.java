package com.github.maximtereshchenko.snapdragon;

record NextEpoch(
    Epoch epoch,
    EpochTrainingStatistics epochTrainingStatistics,
    EpochValidationStatistics epochValidationStatistics
) implements TrainingResult {}
