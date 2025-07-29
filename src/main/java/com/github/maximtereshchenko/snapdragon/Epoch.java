package com.github.maximtereshchenko.snapdragon;

import java.util.ArrayList;
import java.util.function.BiFunction;

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
        var trainingDatasetConsumingResult = datasetConsumingResult(
            neuralNetwork,
            trainingDataset,
            (currentNeuralNetwork, batchedLabeledSample) -> currentNeuralNetwork.calibrated(
                batchedLabeledSample.inputs(),
                batchedLabeledSample.labels(),
                lossFunction,
                learningRate
            )
        );
        var validationStatistics = datasetConsumingResult(
            trainingDatasetConsumingResult.neuralNetwork(),
            validationDataset,
            (currentNeuralNetwork, batchedLabeledSample) -> currentNeuralNetwork
        )
                                       .neuralNetworkStatistics();
        var epochStatistics = new EpochStatistics(
            trainingDatasetConsumingResult.neuralNetworkStatistics(),
            validationStatistics
        );
        return switch (patience.next(validationStatistics.averageLoss())) {
            case Improvement(var nextPatience) -> nextEpoch(
                nextPatience,
                trainingDatasetConsumingResult.neuralNetwork(),
                epochStatistics
            );
            case NoImprovement(var nextPatience) -> nextEpoch(
                nextPatience,
                neuralNetwork,
                epochStatistics
            );
            case Stop() -> new EarlyStop(neuralNetwork, epochStatistics);
        };
    }

    private DatasetConsumingResult datasetConsumingResult(
        NeuralNetwork initialNeuralNetwork,
        Dataset dataset,
        BiFunction<NeuralNetwork, BatchedLabeledSample, NeuralNetwork> nextNeuralNetworkFunction
    ) {
        var currentNeuralNetwork = initialNeuralNetwork;
        var samples = dataset.batchedLabeledSamples();
        var lossesPerBatch = new ArrayList<Double>();
        var accuracySum = 0.0;
        var batches = 0;
        while (samples.hasNext()) {
            var batch = samples.next();
            var outputs = currentNeuralNetwork.outputs(batch.inputs());
            lossesPerBatch.add(loss(outputs, batch.labels()));
            accuracySum += accuracy(outputs, batch.labels());
            currentNeuralNetwork = nextNeuralNetworkFunction.apply(currentNeuralNetwork, batch);
            batches++;
        }
        if (batches == 0) {
            throw new IllegalStateException();
        }
        return new DatasetConsumingResult(
            currentNeuralNetwork,
            new NeuralNetworkStatistics(lossesPerBatch, accuracySum / batches)
        );
    }

    private NextEpoch nextEpoch(
        Patience nextPatience,
        NeuralNetwork nextNeuralNetwork,
        EpochStatistics epochStatistics
    ) {
        return new NextEpoch(
            new Epoch(
                current + 1,
                max,
                trainingDataset,
                validationDataset,
                lossFunction,
                nextNeuralNetwork,
                learningRate,
                nextPatience
            ),
            epochStatistics
        );
    }

    private double accuracy(Outputs outputs, Labels labels) {
        var correct = 0.0;
        var outputsTensor = outputs.tensor();
        var shape = outputsTensor.shape();
        var labelsTensor = labels.tensor();
        for (var row = 0; row < shape[0]; row++) {
            if (maximumValueIndex(outputsTensor, shape, row) ==
                    maximumValueIndex(labelsTensor, shape, row)) {
                correct++;
            }
        }
        return correct / shape[0];
    }

    private int maximumValueIndex(Tensor tensor, int[] shape, int row) {
        var currentMax = Double.MIN_VALUE;
        var index = -1;
        for (var column = 0; column < shape[shape.length - 1]; column++) {
            var currentValue = tensor.value(row, column);
            if (currentValue > currentMax) {
                currentMax = currentValue;
                index = column;
            }
        }
        return index;
    }

    private double loss(Outputs outputs, Labels labels) {
        var loss = lossFunction.loss(outputs.tensor(), labels.tensor());
        var shape = loss.shape();
        return Tensor.horizontalVector(1.0 / shape[0])
                   .broadcasted(1, shape[0])
                   .contracted(loss)
                   .value(0, 0);
    }

    private record DatasetConsumingResult(
        NeuralNetwork neuralNetwork,
        NeuralNetworkStatistics neuralNetworkStatistics
    ) {}
}
