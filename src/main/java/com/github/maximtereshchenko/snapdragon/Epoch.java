package com.github.maximtereshchenko.snapdragon;

import java.util.List;

final class Epoch {

    private final int current;
    private final int max;
    private final TrainingDataset trainingDataset;
    private final ValidationDataset validationDataset;
    private final LossFunction lossFunction;
    private final NeuralNetwork neuralNetwork;
    private final LearningRate learningRate;
    private final Patience patience;

    private Epoch(
        int current,
        int max,
        TrainingDataset trainingDataset,
        ValidationDataset validationDataset,
        LossFunction lossFunction,
        NeuralNetwork neuralNetwork,
        LearningRate learningRate,
        Patience patience
    ) {
        this.current = current;
        this.max = max;
        this.trainingDataset = trainingDataset;
        this.validationDataset = validationDataset;
        this.lossFunction = lossFunction;
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.patience = patience;
    }

    Epoch(
        int max,
        TrainingDataset trainingDataset,
        ValidationDataset validationDataset,
        LossFunction lossFunction,
        NeuralNetwork neuralNetwork,
        LearningRate learningRate,
        Patience patience
    ) {
        this(
            0,
            max,
            trainingDataset,
            validationDataset,
            lossFunction,
            neuralNetwork,
            learningRate,
            patience
        );
    }

    TrainingResult trainingResult() {
        if (current == max) {
            return new End(neuralNetwork);
        }
        var trainingSamples = trainingDataset.labeledSamples();
        var trainingOutputs = neuralNetwork.outputs(trainingSamples.inputs());
        var trainingLoss = loss(trainingOutputs, trainingSamples.labels());
        var calibrated = calibrated(trainingSamples);
        var validationSamples = validationDataset.labeledSamples();
        var validationOutputs = calibrated.outputs(validationSamples.inputs());
        var validationLoss = loss(validationOutputs, validationSamples.labels());
        var epochTrainingStatistics = new EpochTrainingStatistics(
            List.of(trainingLoss),
            accuracy(trainingOutputs, trainingSamples.labels())
        );
        var epochValidationStatistics = new EpochValidationStatistics(
            validationLoss,
            accuracy(validationOutputs, validationSamples.labels())
        );
        return switch (patience.next(validationLoss)) {
            case Improvement(var next) -> new NextEpoch(
                new Epoch(
                    current + 1,
                    max,
                    trainingDataset,
                    validationDataset,
                    lossFunction,
                    calibrated,
                    learningRate,
                    next
                ),
                epochTrainingStatistics,
                epochValidationStatistics
            );
            case NoImprovement(var next) -> new NextEpoch(
                new Epoch(
                    current + 1,
                    max,
                    trainingDataset,
                    validationDataset,
                    lossFunction,
                    neuralNetwork,
                    learningRate,
                    next
                ),
                epochTrainingStatistics,
                epochValidationStatistics
            );
            case Stop() -> new EarlyStop(
                neuralNetwork,
                epochTrainingStatistics,
                epochValidationStatistics
            );
        };
    }

    private double accuracy(Outputs outputs, Labels labels) {
        var correct = 0.0;
        var outputsMatrix = outputs.matrix();
        var labelsMatrix = labels.matrix();
        for (var row = 0; row < outputsMatrix.rows(); row++) {
            if (maximumValueIndex(outputsMatrix, row) == maximumValueIndex(labelsMatrix, row)) {
                correct++;
            }
        }
        return correct / outputsMatrix.rows();
    }

    private int maximumValueIndex(Matrix matrix, int row) {
        var currentMax = Double.MIN_VALUE;
        var index = -1;
        for (var column = 0; column < matrix.columns(); column++) {
            var currentValue = matrix.value(row, column);
            if (currentValue > currentMax) {
                currentMax = currentValue;
                index = column;
            }
        }
        return index;
    }

    private double loss(Outputs outputs, Labels labels) {
        var loss = lossFunction.loss(outputs.matrix(), labels.matrix());
        return Matrix.horizontalVector(1.0 / loss.rows())
                   .broadcasted(1, loss.rows())
                   .product(loss)
                   .value(0, 0);
    }

    private NeuralNetwork calibrated(LabeledSamples trainingSamples) {
        return neuralNetwork.calibrated(
            trainingSamples.inputs(),
            trainingSamples.labels(),
            lossFunction,
            learningRate
        );
    }
}
