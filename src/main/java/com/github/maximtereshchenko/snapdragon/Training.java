package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;

final class Training {

    private final TrainingDataset trainingDataset;
    private final ValidationDataset validationDataset;
    private final LossFunction lossFunction;
    private final NeuralNetwork neuralNetwork;
    private final LearningRate learningRate;
    private final Patience patience;
    private final int maxEpochs;

    Training(
        TrainingDataset trainingDataset,
        ValidationDataset validationDataset,
        LossFunction lossFunction,
        NeuralNetwork neuralNetwork,
        LearningRate learningRate,
        Patience patience,
        int maxEpochs
    ) {
        this.trainingDataset = trainingDataset;
        this.validationDataset = validationDataset;
        this.lossFunction = lossFunction;
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.patience = patience;
        this.maxEpochs = maxEpochs;
    }

    CompletedTraining completedTraining() {
        var current = new Epoch(
            maxEpochs,
            trainingDataset,
            validationDataset,
            lossFunction,
            neuralNetwork,
            learningRate,
            patience
        );
        var epochStatistics = new ArrayList<EpochStatistics>();
        while (true) {
            switch (current.trainingResult()) {
                case NextEpoch(var next, var statistics) -> {
                    epochStatistics.add(statistics);
                    current = next;
                }
                case EarlyStop(var trained, var statistics) -> {
                    epochStatistics.add(statistics);
                    return new CompletedTraining(trained, new Statistics(epochStatistics));
                }
                case End(var trained) -> {
                    return new CompletedTraining(trained, new Statistics(epochStatistics));
                }
            }
        }
    }
}
