package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class TrainingTests {

    @Test
    void givenPatience0_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration1() {
        assertThat(trainedNeuralNetwork(0, 10, 1.0, 1.0))
            .isEqualTo(new FakeNeuralNetwork(1, 0));
    }

    @Test
    void givenImprovingNeuralNetwork_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration2() {
        assertThat(trainedNeuralNetwork(0, 10, 0.2, 0.1, 0.1))
            .isEqualTo(new FakeNeuralNetwork(2, 0));
    }

    @Test
    void given1NoImprovementEpochForPatience1_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration2() {
        assertThat(trainedNeuralNetwork(1, 10, 0.2, 0.1, 0.1, 0.1))
            .isEqualTo(new FakeNeuralNetwork(2, 1));
    }

    @Test
    void givenImprovingNeuralNetworkWithLimit_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration3() {
        assertThat(trainedNeuralNetwork(10, 3, 0.3, 0.2, 0.1))
            .isEqualTo(new FakeNeuralNetwork(3, 1));
    }

    private NeuralNetwork trainedNeuralNetwork(int patience, int epochs, Double... losses) {
        var doubles = new double[]{0};
        return new Training(
            new TrainingDataset(List.of(new StaticLabeledSample(doubles, doubles))),
            new ControlDataset(List.of(new StaticLabeledSample(doubles, doubles))),
            new FakeLossFunction(losses),
            new FakeNeuralNetwork(),
            new LearningRate(1),
            new Patience(patience),
            epochs
        )
                   .trainedNeuralNetwork();
    }
}