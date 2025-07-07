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
                1,
                new FakeLossFunction(1.0, 1.0),
                new FakeNeuralNetwork(),
                10
            )
                .trainedNeuralNetwork()
        )
            .isEqualTo(new FakeNeuralNetwork(1, 0));
    }

    @Test
    void givenImprovingNeuralNetwork_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration2() {
        assertThat(
            new Training(
                Map.of(new double[]{1}, new double[]{1}),
                Map.of(new double[]{1}, new double[]{1}),
                0,
                1,
                new FakeLossFunction(0.2, 0.1, 0.1),
                new FakeNeuralNetwork(),
                10
            )
                .trainedNeuralNetwork()
        )
            .isEqualTo(new FakeNeuralNetwork(2, 0));
    }

    @Test
    void given1NoImprovementEpochForPatience1_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration2() {
        assertThat(
            new Training(
                Map.of(new double[]{1}, new double[]{1}),
                Map.of(new double[]{1}, new double[]{1}),
                1,
                1,
                new FakeLossFunction(0.2, 0.1, 0.1, 0.1),
                new FakeNeuralNetwork(),
                10
            )
                .trainedNeuralNetwork()
        )
            .isEqualTo(new FakeNeuralNetwork(2, 1));
    }

    @Test
    void givenImprovingNeuralNetworkWithLimit_whenTrainedNeuralNetwork_thenNeuralNetworkGeneration3() {
        assertThat(
            new Training(
                Map.of(new double[]{1}, new double[]{1}),
                Map.of(new double[]{1}, new double[]{1}),
                10,
                1,
                new FakeLossFunction(0.3, 0.2, 0.1),
                new FakeNeuralNetwork(),
                3
            )
                .trainedNeuralNetwork()
        )
            .isEqualTo(new FakeNeuralNetwork(3, 1));
    }
}