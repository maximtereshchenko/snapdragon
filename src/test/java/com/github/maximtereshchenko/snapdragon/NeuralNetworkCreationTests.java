package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class NeuralNetworkCreationTests extends BaseNeuralNetworkTest {

    @Test
    void givenNoWeightsBiases_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var empty = List.<Tensor>of();
        assertThatThrownBy(() -> neuralNetwork(empty, empty))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenDifferentSizeWeightsAndBiases_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var weights = List.<Tensor>of();
        var biases = List.of(Tensor.horizontalVector(1));
        assertThatThrownBy(() -> neuralNetwork(weights, biases))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenDifferentWeightColumnsBiasesColumns_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var twoWeights = List.of(Tensor.horizontalVector(1, 2));
        var oneBias = List.of(Tensor.horizontalVector(1));
        assertThatThrownBy(() -> neuralNetwork(twoWeights, oneBias))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenDifferentWeightRowsBiasesColumns_whenCreateNeuralNetwork_thenIllegalArgumentExceptionThrown() {
        var oneWeightOutputLayer = List.of(
            Tensor.horizontalVector(1, 2),
            Tensor.horizontalVector(1)
        );
        var oneBiasOutputLayer = List.of(
            Tensor.horizontalVector(1, 2),
            Tensor.horizontalVector(1)
        );
        assertThatThrownBy(() -> neuralNetwork(oneWeightOutputLayer, oneBiasOutputLayer))
            .isInstanceOf(IllegalArgumentException.class);
    }
}
