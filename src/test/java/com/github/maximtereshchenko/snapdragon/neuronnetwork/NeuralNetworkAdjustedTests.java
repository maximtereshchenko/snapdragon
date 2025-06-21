package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class NeuralNetworkAdjustedTests extends BaseNeuralNetworkTest {

    @Test
    void givenSingleInputOutput_whenAdjusted_thenAdjustedNeuralNetwork() {
        var output = sigmoid(3 * 1 + 2);
        var outputDelta = categoricalCrossEntropyDerivative(0.5, output) *
                              sigmoidDerivative(output);
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(1)),
                List.of(Matrix.horizontalVector(2))
            )
                .adjusted(
                    Matrix.horizontalVector(3),
                    Matrix.horizontalVector(0.5),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(Matrix.horizontalVector(1 - 0.1 * 3 * outputDelta)),
                    List.of(Matrix.horizontalVector(2 - 0.1 * outputDelta))
                )
            );
    }

    @Test
    void givenBatchedInputs_whenAdjusted_thenAdjustedNeuralNetwork() {
        var firstOutput = sigmoid(3 * 1 + 2);
        var secondOutput = sigmoid(4 * 1 + 2);
        var firstOutputDelta = categoricalCrossEntropyDerivative(0.3, firstOutput) *
                                   sigmoidDerivative(firstOutput);
        var secondOutputDelta = categoricalCrossEntropyDerivative(0.7, secondOutput) *
                                    sigmoidDerivative(secondOutput);
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(1)),
                List.of(Matrix.horizontalVector(2))
            )
                .adjusted(
                    Matrix.verticalVector(3, 4),
                    Matrix.verticalVector(0.3, 0.7),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(
                            1 - 0.1 * (3 * firstOutputDelta + 4 * secondOutputDelta) / 2
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            2 - 0.1 * (firstOutputDelta + secondOutputDelta) / 2
                        )
                    )
                )
            );
    }

    @Test
    void givenSingleHiddenNeuron_whenAdjusted_thenAdjustedNeuralNetwork() {
        var hiddenNeuronOutput = sigmoid(5 * 1 + 3);
        var output = sigmoid(hiddenNeuronOutput * 2 + 4);
        var outputDelta = categoricalCrossEntropyDerivative(0.5, output) *
                              sigmoidDerivative(output);
        var hiddenNeuronDelta = 2 * outputDelta * sigmoidDerivative(hiddenNeuronOutput);
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
                .adjusted(
                    Matrix.horizontalVector(5),
                    Matrix.horizontalVector(0.5),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(1 - 0.1 * 5 * hiddenNeuronDelta),
                        Matrix.horizontalVector(2 - 0.1 * hiddenNeuronOutput * outputDelta)
                    ),
                    List.of(
                        Matrix.horizontalVector(3 - 0.1 * hiddenNeuronDelta),
                        Matrix.horizontalVector(4 - 0.1 * outputDelta)
                    )
                )
            );
    }

    @Test
    void givenMultipleInputs_whenAdjusted_thenAdjustedNeuralNetwork() {
        var output = sigmoid(4 * 1 + 5 * 2 + 3);
        var outputDelta = categoricalCrossEntropyDerivative(0.5, output) *
                              sigmoidDerivative(output);
        assertThat(
            neuralNetwork(
                List.of(Matrix.verticalVector(1, 2)),
                List.of(Matrix.horizontalVector(3))
            )
                .adjusted(
                    Matrix.horizontalVector(4, 5),
                    Matrix.horizontalVector(0.5),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.verticalVector(
                            1 - 0.1 * 4 * outputDelta,
                            2 - 0.1 * 5 * outputDelta
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(3 - 0.1 * outputDelta)
                    )
                )
            );
    }

    private double categoricalCrossEntropyDerivative(double label, double output) {
        return -label / output;
    }

    private double sigmoidDerivative(double output) {
        return output * (1 - output);
    }
}
