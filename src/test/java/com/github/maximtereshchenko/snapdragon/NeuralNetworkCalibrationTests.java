package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class NeuralNetworkCalibrationTests extends BaseNeuralNetworkTest {

    @Test
    void givenSingleInputOutput_whenCalibrated_thenCalibratedNeuralNetwork() {
        var output = 0.4 * 0.2 + 0.3;
        assertThat(
            neuralNetwork(
                List.of(Tensor.horizontalVector(0.2)),
                List.of(Tensor.horizontalVector(0.3))
            )
                .calibrated(
                    new Inputs(Tensor.horizontalVector(0.4)),
                    new Labels(Tensor.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.5)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(Tensor.horizontalVector(0.2 - 0.5 * 0.4 * output)),
                    List.of(Tensor.horizontalVector(0.3 - 0.5 * output))
                )
            );
    }

    @Test
    void givenBatchedInputs_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstOutput = 0.4 * 0.2 + 0.3;
        var secondOutput = 0.5 * 0.2 + 0.3;
        assertThat(
            neuralNetwork(
                List.of(Tensor.horizontalVector(0.2)),
                List.of(Tensor.horizontalVector(0.3))
            )
                .calibrated(
                    new Inputs(Tensor.verticalVector(0.4, 0.5)),
                    new Labels(Tensor.verticalVector(0, 0)),
                    new FakeLossFunction(),
                    new LearningRate(0.6)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Tensor.horizontalVector(
                            0.2 - 0.6 * (0.4 * firstOutput + 0.5 * secondOutput) / 2
                        )
                    ),
                    List.of(
                        Tensor.horizontalVector(
                            0.3 - 0.6 * (firstOutput + secondOutput) / 2
                        )
                    )
                )
            );
    }

    @Test
    void givenMultipleInputs_whenCalibrated_thenCalibratedNeuralNetwork() {
        var output = 0.7 * 0.4 + 0.10 * 0.5 + 0.6;
        assertThat(
            neuralNetwork(
                List.of(Tensor.verticalVector(0.4, 0.5)),
                List.of(Tensor.horizontalVector(0.6))
            )
                .calibrated(
                    new Inputs(Tensor.horizontalVector(0.7, 0.10)),
                    new Labels(Tensor.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.12)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Tensor.verticalVector(
                            0.4 - 0.12 * 0.7 * output,
                            0.5 - 0.12 * 0.10 * output
                        )
                    ),
                    List.of(
                        Tensor.horizontalVector(0.6 - 0.12 * output)
                    )
                )
            );
    }

    @Test
    void givenMultipleOutputs_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstOutput = 0.5 * 0.1 + 0.3;
        var secondOutput = 0.5 * 0.2 + 0.4;
        assertThat(
            neuralNetwork(
                List.of(Tensor.horizontalVector(0.1, 0.2)),
                List.of(Tensor.horizontalVector(0.3, 0.4))
            )
                .calibrated(
                    new Inputs(Tensor.horizontalVector(0.5)),
                    new Labels(Tensor.horizontalVector(0, 0)),
                    new FakeLossFunction(),
                    new LearningRate(0.6)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Tensor.horizontalVector(
                            0.1 - 0.6 * 0.5 * firstOutput,
                            0.2 - 0.6 * 0.5 * secondOutput
                        )
                    ),
                    List.of(
                        Tensor.horizontalVector(
                            0.3 - 0.6 * firstOutput,
                            0.4 - 0.6 * secondOutput
                        )
                    )
                )
            );
    }

    @Test
    void givenSingleHiddenNeuron_whenCalibrated_thenCalibratedNeuralNetwork() {
        var hiddenNeuronOutput = 0.5 * 0.1 + 0.3;
        var output = hiddenNeuronOutput * 0.2 + 0.4;
        var hiddenNeuronDelta = 0.2 * output;
        assertThat(
            neuralNetwork(
                List.of(
                    Tensor.horizontalVector(0.1),
                    Tensor.horizontalVector(0.2)
                ),
                List.of(
                    Tensor.horizontalVector(0.3),
                    Tensor.horizontalVector(0.4)
                )
            )
                .calibrated(
                    new Inputs(Tensor.horizontalVector(0.5)),
                    new Labels(Tensor.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.6)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Tensor.horizontalVector(0.1 - 0.6 * 0.5 * hiddenNeuronDelta),
                        Tensor.horizontalVector(0.2 - 0.6 * hiddenNeuronOutput * output)
                    ),
                    List.of(
                        Tensor.horizontalVector(0.3 - 0.6 * hiddenNeuronDelta),
                        Tensor.horizontalVector(0.4 - 0.6 * output)
                    )
                )
            );
    }

    @Test
    void givenMultipleHiddenNeurons_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstHiddenNeuronOutput = 0.9 * 0.2 + 0.6;
        var secondHiddenNeuronOutput = 0.9 * 0.3 + 0.7;
        var output = firstHiddenNeuronOutput * 0.4 + secondHiddenNeuronOutput * 0.5 + 0.8;
        var firstHiddenNeuronDelta = 0.4 * output;
        var secondHiddenNeuronDelta = 0.5 * output;
        assertThat(
            neuralNetwork(
                List.of(
                    Tensor.horizontalVector(0.2, 0.3),
                    Tensor.verticalVector(0.4, 0.5)
                ),
                List.of(
                    Tensor.horizontalVector(0.6, 0.7),
                    Tensor.horizontalVector(0.8)
                )
            )
                .calibrated(
                    new Inputs(Tensor.horizontalVector(0.9)),
                    new Labels(Tensor.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(1.0)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Tensor.horizontalVector(
                            0.2 - 1.0 * 0.9 * firstHiddenNeuronDelta,
                            0.3 - 1.0 * 0.9 * secondHiddenNeuronDelta
                        ),
                        Tensor.verticalVector(
                            0.4 - 1.0 * firstHiddenNeuronOutput * output,
                            0.5 - 1.0 * secondHiddenNeuronOutput * output
                        )
                    ),
                    List.of(
                        Tensor.horizontalVector(
                            0.6 - 1.0 * firstHiddenNeuronDelta,
                            0.7 - 1.0 * secondHiddenNeuronDelta
                        ),
                        Tensor.horizontalVector(0.8 - 1.0 * output)
                    )
                )
            );
    }

    @Test
    void givenMultipleHiddenLayers_whenCalibrated_thenCalibratedNeuralNetwork() {
        var firstHiddenLayerNeuronOutput = 0.7 * 0.1 + 0.4;
        var secondHiddenLayerNeuronOutput = firstHiddenLayerNeuronOutput * 0.2 + 0.5;
        var output = secondHiddenLayerNeuronOutput * 0.3 + 0.6;
        var secondHiddenLayerNeuronDelta = 0.3 * output;
        var firstHiddenLayerNeuronDelta = 0.2 * secondHiddenLayerNeuronDelta;
        assertThat(
            neuralNetwork(
                List.of(
                    Tensor.horizontalVector(0.1),
                    Tensor.horizontalVector(0.2),
                    Tensor.horizontalVector(0.3)
                ),
                List.of(
                    Tensor.horizontalVector(0.4),
                    Tensor.horizontalVector(0.5),
                    Tensor.horizontalVector(0.6)
                )
            )
                .calibrated(
                    new Inputs(Tensor.horizontalVector(0.7)),
                    new Labels(Tensor.horizontalVector(0)),
                    new FakeLossFunction(),
                    new LearningRate(0.8)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Tensor.horizontalVector(0.1 - 0.8 * 0.7 * firstHiddenLayerNeuronDelta),
                        Tensor.horizontalVector(0.2 - 0.8 * firstHiddenLayerNeuronOutput * secondHiddenLayerNeuronDelta),
                        Tensor.horizontalVector(0.3 - 0.8 * secondHiddenLayerNeuronOutput * output)
                    ),
                    List.of(
                        Tensor.horizontalVector(0.4 - 0.8 * firstHiddenLayerNeuronDelta),
                        Tensor.horizontalVector(0.5 - 0.8 * secondHiddenLayerNeuronDelta),
                        Tensor.horizontalVector(0.6 - 0.8 * output)
                    )
                )
            );
    }

    @Test
    void givenMultipleLayersMultipleNeurons_whenCalibrated_thenCalibratedNeuralNetwork() {
        var output13 = 0.42 * 0.3 + 0.44 * 0.1 + 0.15;
        var output14 = 0.42 * 0.4 + 0.44 * 0.6 + 0.16;
        var output15 = output13 * 0.7 + output14 * 0.9 + 0.17;
        var output16 = output13 * 0.8 + output14 * 1.0 + 0.18;
        var output17 = output15 * 0.11 + output16 * 0.26 + 0.19;
        var output18 = output15 * 0.24 + output16 * 0.14 + 0.20;
        var outputDelta15 = 0.11 * output17 + 0.24 * output18;
        var outputDelta16 = 0.26 * output17 + 0.14 * output18;
        var outputDelta13 = 0.7 * outputDelta15 + 0.8 * outputDelta16;
        var outputDelta14 = 0.9 * outputDelta15 + 1.0 * outputDelta16;
        assertThat(
            neuralNetwork(
                List.of(
                    Tensor.from(List.of(2, 2), 0.3, 0.4, 0.1, 0.6),
                    Tensor.from(List.of(2, 2), 0.7, 0.8, 0.9, 1.0),
                    Tensor.from(List.of(2, 2), 0.11, 0.24, 0.26, 0.14)
                ),
                List.of(
                    Tensor.horizontalVector(0.15, 0.16),
                    Tensor.horizontalVector(0.17, 0.18),
                    Tensor.horizontalVector(0.19, 0.20)
                )
            )
                .calibrated(
                    new Inputs(Tensor.horizontalVector(0.42, 0.44)),
                    new Labels(Tensor.horizontalVector(0, 0)),
                    new FakeLossFunction(),
                    new LearningRate(0.23)
                )
        )
            .isEqualTo(
                neuralNetwork(
                    List.of(
                        Tensor.from(
                            List.of(2, 2),
                            0.3 - 0.23 * 0.42 * outputDelta13,
                            0.4 - 0.23 * 0.42 * outputDelta14,
                            0.1 - 0.23 * 0.44 * outputDelta13,
                            0.6 - 0.23 * 0.44 * outputDelta14
                        ),
                        Tensor.from(
                            List.of(2, 2),
                            0.7 - 0.23 * output13 * outputDelta15,
                            0.8 - 0.23 * output13 * outputDelta16,
                            0.9 - 0.23 * output14 * outputDelta15,
                            1.0 - 0.23 * output14 * outputDelta16
                        ),
                        Tensor.from(
                            List.of(2, 2),
                            0.11 - 0.23 * output15 * output17,
                            0.24 - 0.23 * output15 * output18,
                            0.26 - 0.23 * output16 * output17,
                            0.14 - 0.23 * output16 * output18
                        )
                    ),
                    List.of(
                        Tensor.horizontalVector(
                            0.15 - 0.23 * outputDelta13,
                            0.16 - 0.23 * outputDelta14
                        ),
                        Tensor.horizontalVector(
                            0.17 - 0.23 * outputDelta15,
                            0.18 - 0.23 * outputDelta16
                        ),
                        Tensor.horizontalVector(
                            0.19 - 0.23 * output17,
                            0.20 - 0.23 * output18
                        )
                    )
                )
            );
    }
}