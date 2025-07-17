package com.github.maximtereshchenko.snapdragon;

record NextEpoch(Epoch epoch, EpochStatistics epochStatistics) implements TrainingResult {}
