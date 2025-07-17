package com.github.maximtereshchenko.snapdragon;

import java.util.List;

record NeuralNetworkStatistics(List<Double> lossesPerBatch, double accuracy) {

    double averageLoss() {
        return lossesPerBatch.stream()
                   .mapToDouble(loss -> loss)
                   .average()
                   .orElse(0);
    }
}
