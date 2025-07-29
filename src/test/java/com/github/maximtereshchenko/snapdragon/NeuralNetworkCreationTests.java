package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
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

    @Test
    void givenSingleInputOutput_whenRandomNeuralNetwork_thenNoHiddenLayers() {
        assertThat(random(1, 0, 1))
            .isEqualTo(
                new MultiLayerPerceptron(
                    new NetworkLayers(
                        new InputLayer(1),
                        List.of(),
                        new OutputLayer(
                            new LayerIndex(1),
                            new Biases(Tensor.horizontalVector(0.24053641567148587)),
                            new FakeActivationFunction()
                        )
                    ),
                    new NetworkWeights()
                        .with(
                            new LayerIndex(0),
                            new LayerIndex(1),
                            new Weights(Tensor.horizontalVector(0.730967787376657))
                        )
                )
            );
    }

    @Test
    void givenSingleHiddenLayer_whenRandomNeuralNetwork_thenHiddenNeuronsTwoThirdsInputOutputSum() {
        assertThat(random(1, 1, 2))
            .isEqualTo(
                new MultiLayerPerceptron(
                    new NetworkLayers(
                        new InputLayer(1),
                        List.of(
                            new HiddenLayer(
                                new LayerIndex(1),
                                new Biases(
                                    Tensor.horizontalVector(
                                        0.8791825178724801,
                                        0.9412491794821144
                                    )
                                ),
                                new FakeActivationFunction()
                            )
                        ),
                        new OutputLayer(
                            new LayerIndex(2),
                            new Biases(
                                Tensor.horizontalVector(
                                    0.3851891847407185,
                                    0.984841540199809
                                )
                            ),
                            new FakeActivationFunction()
                        )
                    ),
                    new NetworkWeights()
                        .with(
                            new LayerIndex(0),
                            new LayerIndex(1),
                            new Weights(
                                Tensor.horizontalVector(
                                    0.730967787376657,
                                    0.24053641567148587
                                )
                            )
                        )
                        .with(
                            new LayerIndex(1),
                            new LayerIndex(2),
                            new Weights(
                                Tensor.matrix(
                                    2, 2,
                                    0.6374174253501083, 0.5504370051176339,
                                    0.5975452777972018, 0.3332183994766498
                                )
                            )
                        )
                )
            );
    }
}
