package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

final class TrainingTests {

    @Test
    void givenPatience0_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration1() {
        assertThat(
            new Training(
                Map.of(new double[]{1}, new double[]{1}),
                Map.of(new double[]{1}, new double[]{1}),
                0,
                0.1,
                new CategoricalCrossEntropy(),
                new FakeNeuralNetwork()
            )
                .trainedNeuralNetwork()
        )
            .isEqualTo(new FakeNeuralNetwork(1));
    }

    @Test
    void givenImprovingNeuralNetwork_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration2() {
        assertThat(
            new Training(
                Map.of(new double[]{1}, new double[]{1}),
                Map.of(new double[]{1}, new double[]{1}),
                0,
                0.1,
                new CategoricalCrossEntropy(),
                new FakeNeuralNetwork(0.0, 0.9)
            )
                .trainedNeuralNetwork()
        )
            .isEqualTo(new FakeNeuralNetwork(2));
    }
}