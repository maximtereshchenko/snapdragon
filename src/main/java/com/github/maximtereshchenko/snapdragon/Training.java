package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;

public final class Training {

    private final Map<double[], double[]> trainingDataset;
    private final Map<double[], double[]> controlDataset;
    private final int patience;
    private final double learningRate;
    private final LossFunction lossFunction;
    private final NeuralNetwork neuralNetwork;
    private final int total;

    public Training(
        Map<double[], double[]> trainingDataset,
        Map<double[], double[]> controlDataset,
        int patience,
        double learningRate,
        LossFunction lossFunction,
        NeuralNetwork neuralNetwork,
        int total
    ) {
        this.trainingDataset = trainingDataset;
        this.controlDataset = controlDataset;
        this.patience = patience;
        this.learningRate = learningRate;
        this.lossFunction = lossFunction;
        this.neuralNetwork = neuralNetwork;
        this.total = total;
    }

    NeuralNetwork trainedNeuralNetwork() {
        var bestLoss = Double.MAX_VALUE;
        var bestNeuralNetwork = neuralNetwork;
        var noImprovementEpochs = 0;
        var epochs = 0;
        do {
            var entries = new ArrayList<>(trainingDataset.entrySet());
            Collections.shuffle(entries);
            var currentNeuralNetwork = bestNeuralNetwork;
            for (var entry : entries) {
                currentNeuralNetwork = currentNeuralNetwork.adjusted(
                    Matrix.horizontalVector(entry.getKey()),
                    Matrix.horizontalVector(entry.getValue()),
                    lossFunction,
                    learningRate
                );
            }
            var inputs = new double[controlDataset.size()][];
            var labels = new double[controlDataset.size()][];
            var controlEntries = new ArrayList<>(controlDataset.entrySet());
            for (var i = 0; i < controlEntries.size(); i++) {
                inputs[i] = controlEntries.get(i).getKey();
                labels[i] = controlEntries.get(i).getValue();
            }
            var prediction = currentNeuralNetwork.prediction(Matrix.from(inputs));
            var loss = lossFunction.loss(prediction, Matrix.from(labels));
            var epochLoss = 0.0;
            for (var i = 0; i < loss.rows(); i++) {
                epochLoss += loss.value(i, 0);
            }
            epochLoss /= loss.rows();
            if (epochLoss < bestLoss) {
                noImprovementEpochs = 0;
                bestLoss = epochLoss;
                bestNeuralNetwork = currentNeuralNetwork;
            } else {
                noImprovementEpochs++;
            }
            epochs++;
        } while (noImprovementEpochs <= patience && epochs < total);
        return bestNeuralNetwork;
    }
}
