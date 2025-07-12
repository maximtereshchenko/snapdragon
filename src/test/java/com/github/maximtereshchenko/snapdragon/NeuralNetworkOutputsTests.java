package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class NeuralNetworkOutputsTests extends BaseNeuralNetworkTest {

    @Test
    void givenDifferentSizeInputs_whenOutputs_thenIllegalArgumentExceptionThrown() {
        var neuralNetwork = neuralNetwork(
            List.of(Matrix.horizontalVector(0.1)),
            List.of(Matrix.horizontalVector(0.1))
        );
        var tooManyInputs = new Inputs(Matrix.horizontalVector(0.1, 0.2));
        assertThatThrownBy(() -> neuralNetwork.outputs(tooManyInputs))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenSingleInputOutput_whenPrediction_thenExpectedOutputs() {
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(0.1)),
                List.of(Matrix.horizontalVector(0.2))
            )
                .outputs(new Inputs(Matrix.horizontalVector(0.3)))
        )
            .isEqualTo(new Outputs(Matrix.horizontalVector(0.3 * 0.1 + 0.2)));
    }

    @Test
    void givenBatchedInputs_whenPrediction_thenExpectedOutputs() {
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(0.1)),
                List.of(Matrix.horizontalVector(0.2))
            )
                .outputs(new Inputs(Matrix.verticalVector(0.3, 0.4)))
        )
            .isEqualTo(
                new Outputs(Matrix.verticalVector(0.3 * 0.1 + 0.2, 0.4 * 0.1 + 0.2))
            );
    }

    @Test
    void givenMultipleInputs_whenPrediction_thenExpectedOutputs() {
        assertThat(
            neuralNetwork(
                List.of(Matrix.verticalVector(0.1, 0.2)),
                List.of(Matrix.horizontalVector(0.3))
            )
                .outputs(new Inputs(Matrix.horizontalVector(0.4, 0.5)))
        )
            .isEqualTo(new Outputs(Matrix.horizontalVector(0.4 * 0.1 + 0.5 * 0.2 + 0.3)));
    }

    @Test
    void givenMultipleOutputs_whenPrediction_thenExpectedOutputs() {
        assertThat(
            neuralNetwork(
                List.of(Matrix.horizontalVector(0.1, 0.2)),
                List.of(Matrix.horizontalVector(0.3, 0.4))
            )
                .outputs(new Inputs(Matrix.horizontalVector(0.5)))
        )
            .isEqualTo(
                new Outputs(Matrix.horizontalVector(0.5 * 0.1 + 0.3, 0.5 * 0.2 + 0.4))
            );
    }

    @Test
    void givenSingleNeuronHiddenLayer_whenPrediction_thenExpectedOutputs() {
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
                .outputs(new Inputs(Matrix.horizontalVector(0.5)))
        )
            .isEqualTo(new Outputs(Matrix.horizontalVector((0.5 * 0.1 + 0.3) * 0.2 + 0.4)));
    }

    @Test
    void givenMultipleNeuronHiddenLayer_whenPrediction_thenExpectedOutputs() {
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.horizontalVector(0.1, 0.2),
                    Matrix.verticalVector(0.3, 0.4)
                ),
                List.of(
                    Matrix.horizontalVector(0.5, 0.6),
                    Matrix.horizontalVector(0.7)
                )
            )
                .outputs(new Inputs(Matrix.horizontalVector(0.8)))
        )
            .isEqualTo(
                new Outputs(
                    Matrix.horizontalVector(
                        (0.8 * 0.1 + 0.5) * 0.3 +
                            (0.8 * 0.2 + 0.6) * 0.4 +
                            0.7
                    )
                )
            );
    }

    @Test
    void givenMultipleHiddenLayers_whenPrediction_thenExpectedOutputs() {
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
                .outputs(new Inputs(Matrix.horizontalVector(0.7)))
        )
            .isEqualTo(
                new Outputs(
                    Matrix.horizontalVector(((0.7 * 0.1 + 0.4) * 0.2 + 0.5) * 0.3 + 0.6)
                )
            );
    }

    @Test
    void givenMultipleLayersMultipleNeurons_whenPrediction_thenExpectedOutputs() {
        var neuron13 = 0.19 * 0.1 + 0.20 * 0.3 + 0.13;
        var neuron14 = 0.19 * 0.2 + 0.20 * 0.4 + 0.14;
        var neuron15 = neuron13 * 0.5 + neuron14 * 0.7 + 0.15;
        var neuron16 = neuron13 * 0.6 + neuron14 * 0.8 + 0.16;
        assertThat(
            neuralNetwork(
                List.of(
                    Matrix.from(new double[][]{{0.1, 0.2}, {0.3, 0.4}}),
                    Matrix.from(new double[][]{{0.5, 0.6}, {0.7, 0.8}}),
                    Matrix.from(new double[][]{{0.9, 0.10}, {0.11, 0.12}})
                ),
                List.of(
                    Matrix.horizontalVector(0.13, 0.14),
                    Matrix.horizontalVector(0.15, 0.16),
                    Matrix.horizontalVector(0.17, 0.18)
                )
            )
                .outputs(
                    new Inputs(Matrix.horizontalVector(0.19, 0.20))
                )
        )
            .isEqualTo(
                new Outputs(
                    Matrix.horizontalVector(
                        neuron15 * 0.9 + neuron16 * 0.11 + 0.17,
                        neuron15 * 0.10 + neuron16 * 0.12 + 0.18
                    )
                )
            );
    }
}
