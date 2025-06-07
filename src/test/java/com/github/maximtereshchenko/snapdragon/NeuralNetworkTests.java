package com.github.maximtereshchenko.snapdragon;

import com.github.maximtereshchenko.snapdragon.api.HiddenLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.api.InputLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.api.NeuralNetworkConfiguration;
import com.github.maximtereshchenko.snapdragon.api.OutputLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.domain.NeuralNetworkFactory;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

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

    @Test
    void givenMultipleNeuronHiddenLayer_whenPredict_thenExpectedPrediction() {
        assertThat(
            prediction(
                new NeuralNetworkConfiguration(
                    new InputLayerConfiguration(
                        1,
                        List.of(
                            List.of(1.0, 2.0)
                        )
                    ),
                    List.of(
                        new HiddenLayerConfiguration(
                            List.of(3.0, 4.0),
                            List.of(
                                List.of(5.0),
                                List.of(6.0)
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(7.0)
                    )
                ),
                new double[]{8}
            )
        )
            .isEqualTo(
                new double[]{
                    sigmoid(
                        sigmoid(8 * 1 + 3) * 5 +
                            sigmoid(8 * 2 + 4) * 6 +
                            7
                    )
                }
            );
    }

    @Test
    void givenMultipleHiddenLayers_whenPredict_thenExpectedPrediction() {
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
                        ),
                        new HiddenLayerConfiguration(
                            List.of(4.0),
                            List.of(
                                List.of(5.0)
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(6.0)
                    )
                ),
                new double[]{7}
            )
        )
            .isEqualTo(
                new double[]{
                    sigmoid(sigmoid(sigmoid(7 * 1 + 2) * 3 + 4) * 5 + 6)
                }
            );
    }

    @Test
    void givenMultipleLayersMultipleNeurons_whenPredict_thenExpectedPrediction() {
        var neuron5 = sigmoid(19 * 1 + 20 * 3 + 5);
        var neuron6 = sigmoid(19 * 2 + 20 * 4 + 6);
        var neuron11 = sigmoid(neuron5 * 7 + neuron6 * 9 + 11);
        var neuron12 = sigmoid(neuron5 * 8 + neuron6 * 10 + 12);
        assertThat(
            prediction(
                new NeuralNetworkConfiguration(
                    new InputLayerConfiguration(
                        2,
                        List.of(
                            List.of(1.0, 2.0),
                            List.of(3.0, 4.0)
                        )
                    ),
                    List.of(
                        new HiddenLayerConfiguration(
                            List.of(5.0, 6.0),
                            List.of(
                                List.of(7.0, 8.0),
                                List.of(9.0, 10.0)
                            )
                        ),
                        new HiddenLayerConfiguration(
                            List.of(11.0, 12.0),
                            List.of(
                                List.of(13.0, 14.0),
                                List.of(15.0, 16.0)
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(17.0, 18.0)
                    )
                ),
                new double[]{19, 20}
            )
        )
            .isEqualTo(
                new double[]{
                    sigmoid(neuron11 * 13 + neuron12 * 15 + 17),
                    sigmoid(neuron11 * 14 + neuron12 * 16 + 18)
                }
            );
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
