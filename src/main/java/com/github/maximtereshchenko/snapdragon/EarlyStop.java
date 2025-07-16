package com.github.maximtereshchenko.snapdragon;

record EarlyStop(
    NeuralNetwork neuralNetwork,
    EpochTrainingStatistics epochTrainingStatistics,
    EpochValidationStatistics epochValidationStatistics
) implements TrainingResult {}
