package com.github.maximtereshchenko.snapdragon;

import java.util.List;

record EpochTrainingStatistics(List<Double> lossesPerBatch, double accuracy) {}
