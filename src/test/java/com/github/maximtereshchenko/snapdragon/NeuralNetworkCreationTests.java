package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class NeuralNetworkCreationTests extends BaseNeuralNetworkTest {

    @Test
    void givenNoWeightsBiases_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var empty = List.<Matrix>of();
        assertThatThrownBy(() -> neuralNetwork(empty, empty))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenDifferentSizeWeightsAndBiases_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var weights = List.<Matrix>of();
        var biases = List.of(Matrix.horizontalVector(1));
        assertThatThrownBy(() -> neuralNetwork(weights, biases))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenDifferentWeightColumnsBiasesColumns_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var twoWeights = List.of(Matrix.horizontalVector(1, 2));
        var oneBias = List.of(Matrix.horizontalVector(1));
        assertThatThrownBy(() -> neuralNetwork(twoWeights, oneBias))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenDifferentWeightRowsBiasesColumns_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var oneWeightOutputLayer = List.of(
            Matrix.horizontalVector(1, 2),
            Matrix.horizontalVector(1)
        );
        var oneBiasOutputLayer = List.of(
            Matrix.horizontalVector(1, 2),
            Matrix.horizontalVector(1)
        );
        assertThatThrownBy(() -> neuralNetwork(oneWeightOutputLayer, oneBiasOutputLayer))
            .isInstanceOf(IllegalArgumentException.class);
    }
}
