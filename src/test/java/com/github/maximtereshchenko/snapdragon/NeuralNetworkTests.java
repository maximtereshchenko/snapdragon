package com.github.maximtereshchenko.snapdragon;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.maximtereshchenko.snapdragon.api.HiddenLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.api.InputLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.api.NeuralNetworkConfiguration;
import com.github.maximtereshchenko.snapdragon.api.OutputLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.domain.NeuralNetworkFactory;
import java.util.List;
import org.junit.jupiter.api.Test;

final class NeuralNetworkTests {

    @Test
    void givenSingleInputOutput_whenPredict_thenExpectedPrediction() {
        assertThat(
            prediction(
                new NeuralNetworkConfiguration(
                    new InputLayerConfiguration(
                        1,
                        List.of(
                            List.of(1.0)
                        )
                    ),
                    List.of(),
                    new OutputLayerConfiguration(
                        List.of(2.0)
                    )
                ),
                new double[]{3}
            )
        )
            .isEqualTo(new double[]{sigmoid(3 * 1 + 2)});
    }

    @Test
    void givenMultipleInputs_whenPredict_thenExpectedPrediction() {
        assertThat(
            prediction(
                new NeuralNetworkConfiguration(
                    new InputLayerConfiguration(
                        2,
                        List.of(
                            List.of(1.0),
                            List.of(2.0)
                        )
                    ),
                    List.of(),
                    new OutputLayerConfiguration(
                        List.of(3.0)
                    )
                ),
                new double[]{4, 5}
            )
        )
            .isEqualTo(new double[]{sigmoid(4 * 1 + 5 * 2 + 3)});
    }

    @Test
    void givenMultipleOutputs_whenPredict_thenExpectedPrediction() {
        assertThat(
            prediction(
                new NeuralNetworkConfiguration(
                    new InputLayerConfiguration(
                        1,
                        List.of(
                            List.of(1.0, 2.0)
                        )
                    ),
                    List.of(),
                    new OutputLayerConfiguration(
                        List.of(3.0, 4.0)
                    )
                ),
                new double[]{5}
            )
        )
            .isEqualTo(
                new double[]{
                    sigmoid(5 * 1 + 3),
                    sigmoid(5 * 2 + 4)
                }
            );
    }

    @Test
    void givenSingleNeuronHiddenLayer_whenPredict_thenExpectedPrediction() {
        assertThat(
            prediction(
                new NeuralNetworkConfiguration(
                    new InputLayerConfiguration(
                        1,
                        List.of(
                            List.of(1.0)
                        )
                    ),
                    List.of(
                        new HiddenLayerConfiguration(
                            List.of(2.0),
                            List.of(
                                List.of(3.0)
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(4.0)
                    )
                ),
                new double[]{5}
            )
        )
            .isEqualTo(new double[]{sigmoid(sigmoid(5 * 1 + 2) * 3 + 4)});
    }

    private double[] prediction(NeuralNetworkConfiguration configuration, double[] inputs) {
        return new NeuralNetworkFactory()
            .neuralNetwork(configuration)
            .prediction(inputs);
    }

    private double sigmoid(double value) {
        return 1 / (1 + Math.pow(Math.E, -value));
    }
}
