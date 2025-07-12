package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class NeuralNetworkCalibrationTests extends BaseNeuralNetworkTest {

    @Test
    void givenSingleInputOutput_whenCalibrated_thenCalibratedNeuralNetwork() {
        var output = 0.4 * 0.2 + 0.3;
        var outputDelta = output * output;
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(0.2)),
                List.of(Matrix.horizontalVector(0.3))
            )
                .calibrated(
                    new Inputs(Matrix.horizontalVector(0.4)),
                    new Labels(Matrix.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.5)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(Matrix.horizontalVector(0.2 - 0.5 * 0.4 * outputDelta)),
                    List.of(Matrix.horizontalVector(0.3 - 0.5 * outputDelta))
                )
            );
    }

    @Test
    void givenBatchedInputs_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstOutput = 0.4 * 0.2 + 0.3;
        var secondOutput = 0.5 * 0.2 + 0.3;
        var firstOutputDelta = firstOutput * firstOutput;
        var secondOutputDelta = secondOutput * secondOutput;
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(0.2)),
                List.of(Matrix.horizontalVector(0.3))
            )
                .calibrated(
                    new Inputs(Matrix.verticalVector(0.4, 0.5)),
                    new Labels(Matrix.verticalVector(0, 0)),
                    new FakeLossFunction(),
                    new LearningRate(0.6)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(
                            0.2 - 0.6 * (0.4 * firstOutputDelta + 0.5 * secondOutputDelta) / 2
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            0.3 - 0.6 * (firstOutputDelta + secondOutputDelta) / 2
                        )
                    )
                )
            );
    }

    @Test
    void givenMultipleInputs_whenCalibrated_thenCalibratedNeuralNetwork() {
        var output = 0.7 * 0.4 + 0.8 * 0.5 + 0.6;
        var outputDelta = output * output;
        assertThat(
            neuralNetwork(
                List.of(Matrix.verticalVector(0.4, 0.5)),
                List.of(Matrix.horizontalVector(0.6))
            )
                .calibrated(
                    new Inputs(Matrix.horizontalVector(0.7, 0.8)),
                    new Labels(Matrix.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.9)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.verticalVector(
                            0.4 - 0.9 * 0.7 * outputDelta,
                            0.5 - 0.9 * 0.8 * outputDelta
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(0.6 - 0.9 * outputDelta)
                    )
                )
            );
    }

    @Test
    void givenMultipleOutputs_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstOutput = 0.5 * 0.1 + 0.3;
        var secondOutput = 0.5 * 0.2 + 0.4;
        var firstOutputDelta = firstOutput * firstOutput;
        var secondOutputDelta = secondOutput * secondOutput;
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(0.1, 0.2)),
                List.of(Matrix.horizontalVector(0.3, 0.4))
            )
                .calibrated(
                    new Inputs(Matrix.horizontalVector(0.5)),
                    new Labels(Matrix.horizontalVector(0, 0)),
                    new FakeLossFunction(),
                    new LearningRate(0.6)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(
                            0.1 - 0.6 * 0.5 * firstOutputDelta,
                            0.2 - 0.6 * 0.5 * secondOutputDelta
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            0.3 - 0.6 * firstOutputDelta,
                            0.4 - 0.6 * secondOutputDelta
                        )
                    )
                )
            );
    }

    @Test
    void givenSingleHiddenNeuron_whenCalibrated_thenCalibratedNeuralNetwork() {
        var hiddenNeuronOutput = 0.5 * 0.1 + 0.3;
        var output = hiddenNeuronOutput * 0.2 + 0.4;
        var outputDelta = output * output;
        var hiddenNeuronDelta = 0.2 * outputDelta * hiddenNeuronOutput;
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.horizontalVector(0.1),
                    Matrix.horizontalVector(0.2)
                ),
                List.of(
                    Matrix.horizontalVector(0.3),
                    Matrix.horizontalVector(0.4)
                )
            )
                .calibrated(
                    new Inputs(Matrix.horizontalVector(0.5)),
                    new Labels(Matrix.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.6)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(0.1 - 0.6 * 0.5 * hiddenNeuronDelta),
                        Matrix.horizontalVector(0.2 - 0.6 * hiddenNeuronOutput * outputDelta)
                    ),
                    List.of(
                        Matrix.horizontalVector(0.3 - 0.6 * hiddenNeuronDelta),
                        Matrix.horizontalVector(0.4 - 0.6 * outputDelta)
                    )
                )
            );
    }

    @Test
    void givenMultipleHiddenNeurons_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstHiddenNeuronOutput = 0.9 * 0.2 + 0.6;
        var secondHiddenNeuronOutput = 0.9 * 0.3 + 0.7;
        var output = firstHiddenNeuronOutput * 0.4 + secondHiddenNeuronOutput * 0.5 + 0.8;
        var outputDelta = output * output;
        var firstHiddenNeuronDelta = 0.4 * outputDelta * firstHiddenNeuronOutput;
        var secondHiddenNeuronDelta = 0.5 * outputDelta * secondHiddenNeuronOutput;
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.horizontalVector(0.2, 0.3),
                    Matrix.verticalVector(0.4, 0.5)
                ),
                List.of(
                    Matrix.horizontalVector(0.6, 0.7),
                    Matrix.horizontalVector(0.8)
                )
            )
                .calibrated(
                    new Inputs(Matrix.horizontalVector(0.9)),
                    new Labels(Matrix.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(1.0)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(
                            0.2 - 1.0 * 0.9 * firstHiddenNeuronDelta,
                            0.3 - 1.0 * 0.9 * secondHiddenNeuronDelta
                        ),
                        Matrix.verticalVector(
                            0.4 - 1.0 * firstHiddenNeuronOutput * outputDelta,
                            0.5 - 1.0 * secondHiddenNeuronOutput * outputDelta
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            0.6 - 1.0 * firstHiddenNeuronDelta,
                            0.7 - 1.0 * secondHiddenNeuronDelta
                        ),
                        Matrix.horizontalVector(0.8 - 1.0 * outputDelta)
                    )
                )
            );
    }

    @Test
    void givenMultipleHiddenLayers_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstHiddenLayerNeuronOutput = 0.7 * 0.1 + 0.4;
        var secondHiddenLayerNeuronOutput = firstHiddenLayerNeuronOutput * 0.2 + 0.5;
        var output = secondHiddenLayerNeuronOutput * 0.3 + 0.6;
        var outputDelta = output * output;
        var secondHiddenLayerNeuronDelta = 0.3 * outputDelta * secondHiddenLayerNeuronOutput;
        var firstHiddenLayerNeuronDelta =
            0.2 * secondHiddenLayerNeuronDelta * firstHiddenLayerNeuronOutput;
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.horizontalVector(0.1),
                    Matrix.horizontalVector(0.2),
                    Matrix.horizontalVector(0.3)
                ),
                List.of(
                    Matrix.horizontalVector(0.4),
                    Matrix.horizontalVector(0.5),
                    Matrix.horizontalVector(0.6)
                )
            )
                .calibrated(
                    new Inputs(Matrix.horizontalVector(0.7)),
                    new Labels(Matrix.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.8)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.horizontalVector(0.1 - 0.8 * 0.7 * firstHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(0.2 - 0.8 * firstHiddenLayerNeuronOutput * secondHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(0.3 - 0.8 * secondHiddenLayerNeuronOutput * outputDelta)
                    ),
                    List.of(
                        Matrix.horizontalVector(0.4 - 0.8 * firstHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(0.5 - 0.8 * secondHiddenLayerNeuronDelta),
                        Matrix.horizontalVector(0.6 - 0.8 * outputDelta)
                    )
                )
            );
    }

    @Test
    void givenMultipleLayersMultipleNeurons_whenCalibrated_thenCalibratedNeuralNetwork() {
        var output13 = 0.21 * 0.3 + 0.22 * 0.5 + 0.15;
        var output14 = 0.21 * 0.4 + 0.22 * 0.6 + 0.16;
        var output15 = output13 * 0.7 + output14 * 0.9 + 0.17;
        var output16 = output13 * 0.8 + output14 * 1.0 + 0.18;
        var output17 = output15 * 0.11 + output16 * 0.13 + 0.19;
        var output18 = output15 * 0.12 + output16 * 0.14 + 0.20;
        var outputDelta17 = output17 * output17;
        var outputDelta18 = output18 * output18;
        var outputDelta15 = (0.11 * outputDelta17 + 0.12 * outputDelta18) * output15;
        var outputDelta16 = (0.13 * outputDelta17 + 0.14 * outputDelta18) * output16;
        var outputDelta13 = (0.7 * outputDelta15 + 0.8 * outputDelta16) * output13;
        var outputDelta14 = (0.9 * outputDelta15 + 1.0 * outputDelta16) * output14;
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.from(new double[][]{{0.3, 0.4}, {0.5, 0.6}}),
                    Matrix.from(new double[][]{{0.7, 0.8}, {0.9, 1.0}}),
                    Matrix.from(new double[][]{{0.11, 0.12}, {0.13, 0.14}})
                ),
                List.of(
                    Matrix.horizontalVector(0.15, 0.16),
                    Matrix.horizontalVector(0.17, 0.18),
                    Matrix.horizontalVector(0.19, 0.20)
                )
            )
                .calibrated(
                    new Inputs(Matrix.horizontalVector(0.21, 0.22)),
                    new Labels(Matrix.horizontalVector(0, 0)),
                    new FakeLossFunction(),
                    new LearningRate(0.23)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Matrix.from(
                            new double[][]{
                                {
                                    0.3 - 0.23 * 0.21 * outputDelta13,
                                    0.4 - 0.23 * 0.21 * outputDelta14
                                },
                                {
                                    0.5 - 0.23 * 0.22 * outputDelta13,
                                    0.6 - 0.23 * 0.22 * outputDelta14
                                }
                            }
                        ),
                        Matrix.from(
                            new double[][]{
                                {
                                    0.7 - 0.23 * output13 * outputDelta15,
                                    0.8 - 0.23 * output13 * outputDelta16
                                },
                                {
                                    0.9 - 0.23 * output14 * outputDelta15,
                                    1.0 - 0.23 * output14 * outputDelta16
                                }
                            }
                        ),
                        Matrix.from(
                            new double[][]{
                                {
                                    0.11 - 0.23 * output15 * outputDelta17,
                                    0.12 - 0.23 * output15 * outputDelta18
                                },
                                {
                                    0.13 - 0.23 * output16 * outputDelta17,
                                    0.14 - 0.23 * output16 * outputDelta18
                                }
                            }
                        )
                    ),
                    List.of(
                        Matrix.horizontalVector(
                            0.15 - 0.23 * outputDelta13,
                            0.16 - 0.23 * outputDelta14
                        ),
                        Matrix.horizontalVector(
                            0.17 - 0.23 * outputDelta15,
                            0.18 - 0.23 * outputDelta16
                        ),
                        Matrix.horizontalVector(
                            0.19 - 0.23 * outputDelta17,
                            0.20 - 0.23 * outputDelta18
                        )
                    )
                )
            );
    }
}