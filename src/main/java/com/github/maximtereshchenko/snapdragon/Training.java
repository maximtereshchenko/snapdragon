package com.github.maximtereshchenko.snapdragon;

import java.util.Map;

public final class Training {

    private final Map<double[], double[]> trainingDataset;
    private final Map<double[], double[]> controlDataset;
    private final int patience;
    private final NeuralNetwork neuralNetwork;

    public Training(
        Map<double[], double[]> trainingDataset,
        Map<double[], double[]> controlDataset,
        int patience,
        NeuralNetwork neuralNetwork
    ) {
        this.trainingDataset = trainingDataset;
        this.controlDataset = controlDataset;
        this.patience = patience;
        this.neuralNetwork = neuralNetwork;
    }

    NeuralNetwork trainedNeuralNetwork() {
        return neuralNetwork;
    }
}
