package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

final class TrainingTests {

    @Test
    void givenPatience0_whenTraining_thenInitialNeuralNetwork() {
        var neuralNetwork = new FakeNeuralNetwork();
        assertThat(
            new Training(
                Map.of(new double[]{1}, new double[]{1}),
                Map.of(new double[]{1}, new double[]{1}),
                0,
                neuralNetwork
            )
                .trainedNeuralNetwork()
        )
            .isEqualTo(neuralNetwork);
    }
}