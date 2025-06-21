package com.github.maximtereshchenko.snapdragon;

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

    @Test
    void givenMultipleOutputs_whenAdjusted_thenAdjustedNeuralNetwork() {
        var firstOutput = sigmoid(5 * 1 + 3);
        var secondOutput = sigmoid(5 * 2 + 4);
        var firstOutputDelta = categoricalCrossEntropyDerivative(0.3, firstOutput) *
                                   sigmoidDerivative(firstOutput);
        var secondOutputDelta = categoricalCrossEntropyDerivative(0.7, secondOutput) *
                                    sigmoidDerivative(secondOutput);
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(1, 2)),
                List.of(Matrix.horizontalVector(3, 4))
            )
                .adjusted(
                    Matrix.horizontalVector(5),
                    Matrix.horizontalVector(0.3, 0.7),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(
                            1 - 0.1 * 5 * firstOutputDelta,
                            2 - 0.1 * 5 * secondOutputDelta
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            3 - 0.1 * firstOutputDelta,
                            4 - 0.1 * secondOutputDelta
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
    void givenMultipleHiddenNeurons_whenAdjusted_thenAdjustedNeuralNetwork() {
        var firstHiddenNeuronOutput = sigmoid(8 * 1 + 5);
        var secondHiddenNeuronOutput = sigmoid(8 * 2 + 6);
        var output = sigmoid(firstHiddenNeuronOutput * 3 + secondHiddenNeuronOutput * 4 + 7);
        var outputDelta = categoricalCrossEntropyDerivative(0.5, output) *
                              sigmoidDerivative(output);
        var firstHiddenNeuronDelta = 3 * outputDelta * sigmoidDerivative(firstHiddenNeuronOutput);
        var secondHiddenNeuronDelta = 4 * outputDelta * sigmoidDerivative(secondHiddenNeuronOutput);
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
                .adjusted(
                    Matrix.horizontalVector(8),
                    Matrix.horizontalVector(0.5),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(
                            1 - 0.1 * 8 * firstHiddenNeuronDelta,
                            2 - 0.1 * 8 * secondHiddenNeuronDelta
                        ),
                        Matrix.verticalVector(
                            3 - 0.1 * firstHiddenNeuronOutput * outputDelta,
                            4 - 0.1 * secondHiddenNeuronOutput * outputDelta
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            5 - 0.1 * firstHiddenNeuronDelta,
                            6 - 0.1 * secondHiddenNeuronDelta
                        ),
                        Matrix.horizontalVector(7 - 0.1 * outputDelta)
                    )
                )
            );
    }

    @Test
    void givenMultipleHiddenLayers_whenAdjusted_thenAdjustedNeuralNetwork() {
        var firstHiddenLayerNeuronOutput = sigmoid(7 * 1 + 4);
        var secondHiddenLayerNeuronOutput = sigmoid(firstHiddenLayerNeuronOutput * 2 + 5);
        var output = sigmoid(secondHiddenLayerNeuronOutput * 3 + 6);
        var outputDelta = categoricalCrossEntropyDerivative(0.5, output) *
                              sigmoidDerivative(output);
        var secondHiddenLayerNeuronDelta = 3 * outputDelta * sigmoidDerivative(secondHiddenLayerNeuronOutput);
        var firstHiddenLayerNeuronDelta = 2 * secondHiddenLayerNeuronDelta *
                                              sigmoidDerivative(firstHiddenLayerNeuronOutput);
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
                .adjusted(
                    Matrix.horizontalVector(7),
                    Matrix.horizontalVector(0.5),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(1 - 0.1 * 7 * firstHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(2 - 0.1 * firstHiddenLayerNeuronOutput * secondHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(3 - 0.1 * secondHiddenLayerNeuronOutput * outputDelta)
                    ),
                    List.of(
                        Matrix.horizontalVector(4 - 0.1 * firstHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(5 - 0.1 * secondHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(6 - 0.1 * outputDelta)
                    )
                )
            );
    }

    @Test
    void givenMultipleLayersMultipleNeurons_whenAdjusted_thenAdjustedNeuralNetwork() {
        var output13 = sigmoid(19 * 1 + 20 * 3 + 13);
        var output14 = sigmoid(19 * 2 + 20 * 4 + 14);
        var output15 = sigmoid(output13 * 5 + output14 * 7 + 15);
        var output16 = sigmoid(output13 * 6 + output14 * 8 + 16);
        var output17 = sigmoid(output15 * 9 + output16 * 10 + 17);
        var output18 = sigmoid(output15 * 11 + output16 * 12 + 18);
        var outputDelta17 = categoricalCrossEntropyDerivative(0.3, output17) *
                                sigmoidDerivative(output17);
        var outputDelta18 = categoricalCrossEntropyDerivative(0.7, output18) *
                                sigmoidDerivative(output18);
        var outputDelta15 = (9 * outputDelta17 + 10 * outputDelta18) * sigmoidDerivative(output15);
        var outputDelta16 = (11 * outputDelta17 + 12 * outputDelta18) * sigmoidDerivative(output16);
        var outputDelta13 = (5 * outputDelta15 + 6 * outputDelta16) * sigmoidDerivative(output13);
        var outputDelta14 = (7 * outputDelta15 + 8 * outputDelta16) * sigmoidDerivative(output14);
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
                .adjusted(
                    Matrix.horizontalVector(19, 20),
                    Matrix.horizontalVector(0.3, 0.7),
                    new CategoricalCrossEntropy(),
                    0.1
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.from(
                            new double[][]{
                                {
                                    1 - 0.1 * 19 * outputDelta13,
                                    2 - 0.1 * 19 * outputDelta14
                                },
                                {
                                    3 - 0.1 * 20 * outputDelta13,
                                    4 - 0.1 * 20 * outputDelta14
                                }
                            }
                        ),
                        Matrix.from(
                            new double[][]{
                                {
                                    5 - 0.1 * output13 * outputDelta15,
                                    6 - 0.1 * output13 * outputDelta16
                                },
                                {
                                    7 - 0.1 * output14 * outputDelta15,
                                    8 - 0.1 * output14 * outputDelta16
                                }
                            }
                        ),
                        Matrix.from(
                            new double[][]{
                                {
                                    9 - 0.1 * output15 * outputDelta17,
                                    10 - 0.1 * output15 * outputDelta18
                                },
                                {
                                    11 - 0.1 * output16 * outputDelta17,
                                    12 - 0.1 * output16 * outputDelta18
                                }
                            }
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            13 - 0.1 * outputDelta13,
                            14 - 0.1 * outputDelta14
                        ),
                        Matrix.horizontalVector(
                            15 - 0.1 * outputDelta15,
                            16 - 0.1 * outputDelta16
                        ),
                        Matrix.horizontalVector(
                            17 - 0.1 * outputDelta17,
                            18 - 0.1 * outputDelta18
                        )
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
