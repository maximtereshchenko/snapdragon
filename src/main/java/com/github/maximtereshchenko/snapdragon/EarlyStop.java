package com.github.maximtereshchenko.snapdragon;

record EarlyStop(NeuralNetwork neuralNetwork, EpochStatistics epochStatistics)
    implements TrainingResult {}
