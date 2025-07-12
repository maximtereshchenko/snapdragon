package com.github.maximtereshchenko.snapdragon;

final class Epoch {

    private final int current;
    private final int max;
    private final TrainingDataset trainingDataset;
    private final ControlDataset controlDataset;
    private final LossFunction lossFunction;
    private final NeuralNetwork neuralNetwork;
    private final LearningRate learningRate;
    private final Patience patience;

    private Epoch(
        int current,
        int max,
        TrainingDataset trainingDataset,
        ControlDataset controlDataset,
        LossFunction lossFunction,
        NeuralNetwork neuralNetwork,
        LearningRate learningRate,
        Patience patience
    ) {
        this.current = current;
        this.max = max;
        this.trainingDataset = trainingDataset;
        this.controlDataset = controlDataset;
        this.lossFunction = lossFunction;
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.patience = patience;
    }

    Epoch(
        int max,
        TrainingDataset trainingDataset,
        ControlDataset controlDataset,
        LossFunction lossFunction,
        NeuralNetwork neuralNetwork,
        LearningRate learningRate,
        Patience patience
    ) {
        this(
            0,
            max,
            trainingDataset,
            controlDataset,
            lossFunction,
            neuralNetwork,
            learningRate,
            patience
        );
    }

    TrainingResult trainingResult() {
        if (current == max) {
            return new TrainedNeuralNetwork(neuralNetwork);
        }
        var calibrated = calibrated();
        var loss = loss(calibrated);
        return switch (patience.next(loss)) {
            case Improvement(var next) -> new NextEpoch(
                new Epoch(
                    current + 1,
                    max,
                    trainingDataset,
                    controlDataset,
                    lossFunction,
                    calibrated,
                    learningRate,
                    next
                )
            );
            case NoImprovement(var next) -> new NextEpoch(
                new Epoch(
                    current + 1,
                    max,
                    trainingDataset,
                    controlDataset,
                    lossFunction,
                    neuralNetwork,
                    learningRate,
                    next
                )
            );
            case Stop() -> new TrainedNeuralNetwork(neuralNetwork);
        };
    }

    private double loss(NeuralNetwork neuralNetwork) {
        var controlSamples = controlDataset.labeledSamples();
        var loss = lossFunction.loss(
            neuralNetwork.outputs(controlSamples.inputs()).matrix(),
            controlSamples.labels().matrix()
        );
        return Matrix.horizontalVector(1.0 / loss.rows())
                   .broadcasted(1, loss.rows())
                   .product(loss)
                   .value(0, 0);
    }

    private NeuralNetwork calibrated() {
        var trainingSamples = trainingDataset.labeledSamples();
        return neuralNetwork.calibrated(
            trainingSamples.inputs(),
            trainingSamples.labels(),
            lossFunction,
            learningRate
        );
    }
}
