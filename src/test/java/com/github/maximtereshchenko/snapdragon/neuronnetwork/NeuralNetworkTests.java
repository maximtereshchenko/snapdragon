package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.neuronnetwork.api.*;
import com.github.maximtereshchenko.snapdragon.neuronnetwork.domain.NeuralNetworkFactory;
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
                        List.of(
                            new InputNeuronConfiguration(
                                List.of(1.0)
                            )
                        )
                    ),
                    List.of(),
                    new OutputLayerConfiguration(
                        List.of(
                            new OutputNeuronConfiguration(2)
                        )
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
                        List.of(
                            new InputNeuronConfiguration(
                                List.of(1.0)
                            ),
                            new InputNeuronConfiguration(
                                List.of(2.0)
                            )
                        )
                    ),
                    List.of(),
                    new OutputLayerConfiguration(
                        List.of(
                            new OutputNeuronConfiguration(3)
                        )
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
                        List.of(
                            new InputNeuronConfiguration(
                                List.of(1.0, 2.0)
                            )
                        )
                    ),
                    List.of(),
                    new OutputLayerConfiguration(
                        List.of(
                            new OutputNeuronConfiguration(3),
                            new OutputNeuronConfiguration(4)
                        )
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
                        List.of(
                            new InputNeuronConfiguration(
                                List.of(1.0)
                            )
                        )
                    ),
                    List.of(
                        new HiddenLayerConfiguration(
                            List.of(
                                new HiddenNeuronConfiguration(
                                    2,
                                    List.of(3.0)
                                )
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(
                            new OutputNeuronConfiguration(4)
                        )
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
                        List.of(
                            new InputNeuronConfiguration(
                                List.of(1.0, 2.0)
                            )
                        )
                    ),
                    List.of(
                        new HiddenLayerConfiguration(
                            List.of(
                                new HiddenNeuronConfiguration(
                                    3,
                                    List.of(4.0)
                                ),
                                new HiddenNeuronConfiguration(
                                    5,
                                    List.of(6.0)
                                )
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(
                            new OutputNeuronConfiguration(7)
                        )
                    )
                ),
                new double[]{8}
            )
        )
            .isEqualTo(
                new double[]{
                    sigmoid(
                        sigmoid(8 * 1 + 3) * 4 +
                            sigmoid(8 * 2 + 5) * 6 +
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
                        List.of(
                            new InputNeuronConfiguration(
                                List.of(1.0)
                            )
                        )
                    ),
                    List.of(
                        new HiddenLayerConfiguration(
                            List.of(
                                new HiddenNeuronConfiguration(
                                    2,
                                    List.of(3.0)
                                )
                            )
                        ),
                        new HiddenLayerConfiguration(
                            List.of(
                                new HiddenNeuronConfiguration(
                                    4,
                                    List.of(5.0)
                                )
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(
                            new OutputNeuronConfiguration(6)
                        )
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
        var neuron8 = sigmoid(19 * 2 + 20 * 4 + 8);
        var neuron11 = sigmoid(neuron5 * 6 + neuron8 * 9 + 11);
        var neuron14 = sigmoid(neuron5 * 7 + neuron8 * 10 + 14);
        assertThat(
            prediction(
                new NeuralNetworkConfiguration(
                    new InputLayerConfiguration(
                        List.of(
                            new InputNeuronConfiguration(
                                List.of(1.0, 2.0)
                            ),
                            new InputNeuronConfiguration(
                                List.of(3.0, 4.0)
                            )
                        )
                    ),
                    List.of(
                        new HiddenLayerConfiguration(
                            List.of(
                                new HiddenNeuronConfiguration(
                                    5,
                                    List.of(6.0, 7.0)
                                ),
                                new HiddenNeuronConfiguration(
                                    8,
                                    List.of(9.0, 10.0)
                                )
                            )
                        ),
                        new HiddenLayerConfiguration(
                            List.of(
                                new HiddenNeuronConfiguration(
                                    11,
                                    List.of(12.0, 13.0)
                                ),
                                new HiddenNeuronConfiguration(
                                    14,
                                    List.of(15.0, 16.0)
                                )
                            )
                        )
                    ),
                    new OutputLayerConfiguration(
                        List.of(
                            new OutputNeuronConfiguration(17),
                            new OutputNeuronConfiguration(18)
                        )
                    )
                ),
                new double[]{19, 20}
            )
        )
            .isEqualTo(
                new double[]{
                    sigmoid(neuron11 * 13 + neuron14 * 15 + 17),
                    sigmoid(neuron11 * 13 + neuron14 * 16 + 18)
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
