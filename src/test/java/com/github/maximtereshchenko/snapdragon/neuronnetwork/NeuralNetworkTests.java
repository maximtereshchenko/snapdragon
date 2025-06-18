package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class NeuralNetworkTests {

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

    @Test
    void givenDifferentSizeInputs_whenOutputs_thenIllegalArgumentExceptionThrown() {
        var neuralNetwork = neuralNetwork(
            List.of(Matrix.horizontalVector(1)),
            List.of(Matrix.horizontalVector(1))
        );
        var tooManyInputs = Matrix.horizontalVector(1, 2);
        assertThatThrownBy(() -> neuralNetwork.outputs(tooManyInputs))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenSingleInputOutput_whenOutputs_thenExpectedPrediction() {
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(1)),
                List.of(Matrix.horizontalVector(2))
            )
                .outputs(Matrix.horizontalVector(3))
        )
            .isEqualTo(Matrix.horizontalVector(sigmoid(3 * 1 + 2)));
    }

    @Test
    void givenMultipleInputs_whenOutputs_thenExpectedPrediction() {
        assertThat(
            neuralNetwork(
                List.of(Matrix.verticalVector(1, 2)),
                List.of(Matrix.horizontalVector(3))
            )
                .outputs(Matrix.horizontalVector(4, 5))
        )
            .isEqualTo(Matrix.horizontalVector(sigmoid(4 * 1 + 5 * 2 + 3)));
    }

    @Test
    void givenMultipleOutputs_whenOutputs_thenExpectedPrediction() {
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(1, 2)),
                List.of(Matrix.horizontalVector(3, 4))
            )
                .outputs(Matrix.horizontalVector(5))
        )
            .isEqualTo(Matrix.horizontalVector(sigmoid(5 * 1 + 3), sigmoid(5 * 2 + 4)));
    }

    @Test
    void givenSingleNeuronHiddenLayer_whenOutputs_thenExpectedPrediction() {
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.horizontalVector(1),
                    Matrix.horizontalVector(2)
                ),
                List.of(
                    Matrix.horizontalVector(3),
                    Matrix.horizontalVector(4)
                )
            )
                .outputs(Matrix.horizontalVector(5))
        )
            .isEqualTo(Matrix.horizontalVector(sigmoid(sigmoid(5 * 1 + 3) * 2 + 4)));
    }

    @Test
    void givenMultipleNeuronHiddenLayer_whenOutputs_thenExpectedPrediction() {
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.horizontalVector(1, 2),
                    Matrix.verticalVector(3, 4)
                ),
                List.of(
                    Matrix.horizontalVector(5, 6),
                    Matrix.horizontalVector(7)
                )
            )
                .outputs(Matrix.horizontalVector(8))
        )
            .isEqualTo(
                Matrix.horizontalVector(
                    sigmoid(
                        sigmoid(8 * 1 + 5) * 3 +
                            sigmoid(8 * 2 + 6) * 4 +
                            7
                    )
                )
            );
    }

    @Test
    void givenMultipleHiddenLayers_whenOutputs_thenExpectedPrediction() {
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.horizontalVector(1),
                    Matrix.horizontalVector(2),
                    Matrix.horizontalVector(3)
                ),
                List.of(
                    Matrix.horizontalVector(4),
                    Matrix.horizontalVector(5),
                    Matrix.horizontalVector(6)
                )
            )
                .outputs(Matrix.horizontalVector(7))
        )
            .isEqualTo(
                Matrix.horizontalVector(
                    sigmoid(sigmoid(sigmoid(7 * 1 + 4) * 2 + 5) * 3 + 6)
                )
            );
    }

    @Test
    void givenMultipleLayersMultipleNeurons_whenOutputs_thenExpectedPrediction() {
        var neuron13 = sigmoid(19 * 1 + 20 * 3 + 13);
        var neuron14 = sigmoid(19 * 2 + 20 * 4 + 14);
        var neuron15 = sigmoid(neuron13 * 5 + neuron14 * 7 + 15);
        var neuron16 = sigmoid(neuron13 * 6 + neuron14 * 8 + 16);
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.from(new double[][]{{1, 2}, {3, 4}}),
                    Matrix.from(new double[][]{{5, 6}, {7, 8}}),
                    Matrix.from(new double[][]{{9, 10}, {11, 12}})
                ),
                List.of(
                    Matrix.horizontalVector(13, 14),
                    Matrix.horizontalVector(15, 16),
                    Matrix.horizontalVector(17, 18)
                )
            )
                .outputs(
                    Matrix.horizontalVector(19, 20)
                )
        )
            .isEqualTo(
                Matrix.horizontalVector(
                    sigmoid(neuron15 * 9 + neuron16 * 11 + 17),
                    sigmoid(neuron15 * 10 + neuron16 * 12 + 18)
                )
            );
    }

    private NeuralNetwork neuralNetwork(List<Matrix> weights, List<Matrix> biases) {
        return NeuralNetwork.from(weights, biases, new Sigmoid(), new Sigmoid());
    }

    private double sigmoid(double value) {
        return 1 / (1 + Math.pow(Math.E, -value));
    }
}
