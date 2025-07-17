package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;

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

        var calibrated = neuralNetwork;
        var trainingSamples = trainingDataset.batchedLabeledSamples();
        var trainingLossesPerBatch = new ArrayList<Double>();
        var trainingAccuracySum = 0.0;
        var trainingBatches = 0;
        while (trainingSamples.hasNext()) {
            var batch = trainingSamples.next();
            var outputs = calibrated.outputs(batch.inputs());
            trainingLossesPerBatch.add(loss(outputs, batch.labels()));
            trainingAccuracySum += accuracy(outputs, batch.labels());
            calibrated = neuralNetwork.calibrated(
                batch.inputs(),
                batch.labels(),
                lossFunction,
                learningRate
            );
            trainingBatches++;
        }
        if (trainingBatches == 0) {
            throw new IllegalStateException();
        }
        var epochTrainingStatistics = new EpochTrainingStatistics(
            trainingLossesPerBatch,
            trainingAccuracySum / trainingBatches
        );

        var validationSamples = validationDataset.batchedLabeledSamples();
        var validationLossesPerBatch = new ArrayList<Double>();
        var validationAccuracySum = 0.0;
        var validationBatches = 0;
        while (validationSamples.hasNext()) {
            var batch = validationSamples.next();
            var outputs = calibrated.outputs(batch.inputs());
            validationLossesPerBatch.add(loss(outputs, batch.labels()));
            validationAccuracySum += accuracy(outputs, batch.labels());
            validationBatches++;
        }
        if (validationBatches == 0) {
            throw new IllegalStateException();
        }
        var epochValidationStatistics = new EpochValidationStatistics(
            validationLossesPerBatch.stream()
                .mapToDouble(loss -> loss)
                .sum() / validationBatches,
            validationAccuracySum / validationBatches
        );
        return switch (patience.next(epochValidationStatistics.loss())) {
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
}
