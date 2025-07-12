package com.github.maximtereshchenko.snapdragon;

final class Training {

    private final TrainingDataset trainingDataset;
    private final ControlDataset controlDataset;
    private final LossFunction lossFunction;
    private final NeuralNetwork neuralNetwork;
    private final LearningRate learningRate;
    private final Patience patience;
    private final int maxEpochs;

    Training(
        TrainingDataset trainingDataset,
        ControlDataset controlDataset,
        LossFunction lossFunction,
        NeuralNetwork neuralNetwork,
        LearningRate learningRate,
        Patience patience,
        int maxEpochs
    ) {
        this.trainingDataset = trainingDataset;
        this.controlDataset = controlDataset;
        this.lossFunction = lossFunction;
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.patience = patience;
        this.maxEpochs = maxEpochs;
    }

    NeuralNetwork trainedNeuralNetwork() {
        var current = new Epoch(
            maxEpochs,
            trainingDataset,
            controlDataset,
            lossFunction,
            neuralNetwork,
            learningRate,
            patience
        );
        while (true) {
            switch (current.trainingResult()) {
                case NextEpoch(var next) -> current = next;
                case TrainedNeuralNetwork(var trained) -> {
                    return trained;
                }
            }
        }
    }
}
