package com.github.maximtereshchenko.snapdragon;

import java.nio.file.Paths;
import java.util.concurrent.ThreadLocalRandom;

final class Main {

    public static void main(String[] args) {
        var completedTraining = new Training(
            new MnistSamples<>(
                Paths.get("./train-images.idx3-ubyte"),
                Paths.get("./train-labels.idx1-ubyte"),
                labeledSamples -> new TrainingDataset(
                    labeledSamples,
                    100,
                    ThreadLocalRandom.current()
                )
            )
                .dataset(),
            new MnistSamples<>(
                Paths.get("./t10k-images.idx3-ubyte"),
                Paths.get("./t10k-labels.idx1-ubyte"),
                labeledSamples -> new ValidationDataset(labeledSamples, Integer.MAX_VALUE)
            )
                .dataset(),
            new CategoricalCrossEntropy(),
            new NeuralNetworkFactory()
                .neuralNetwork(
                    ThreadLocalRandom.current(),
                    28 * 28,
                    2,
                    10,
                    new Sigmoid(),
                    new Softmax()
                ),
            new LearningRate(0.1),
            new Patience(10),
            1
        )
                                    .completedTraining();
        System.out.println(completedTraining.statistics());
    }
}
