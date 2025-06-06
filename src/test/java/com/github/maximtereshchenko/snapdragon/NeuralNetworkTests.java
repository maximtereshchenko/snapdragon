package com.github.maximtereshchenko.snapdragon;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.maximtereshchenko.snapdragon.api.InputLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.api.NeuralNetworkConfiguration;
import com.github.maximtereshchenko.snapdragon.api.OutputLayerConfiguration;
import com.github.maximtereshchenko.snapdragon.domain.NeuralNetwork;
import java.util.List;
import org.junit.jupiter.api.Test;

final class NeuralNetworkTests {

    @Test
    void givenSingleInputOutput_whenPredict_thenExpectedPrediction() {
        var configuration = new NeuralNetworkConfiguration(
            new InputLayerConfiguration(
                1,
                List.of(1.0)
            ),
            List.of(),
            new OutputLayerConfiguration(
                List.of(1.0)
            )
        );
        var neuralNetwork = NeuralNetwork.from(configuration);

        assertThat(neuralNetwork.prediction(new double[]{1}))
            .isEqualTo(
                new double[]{
                    1 / (1 + Math.pow(Math.E, -(1 * 1 + 1)))
                }
            );
    }

    @Test
    void givenMultipleInputs_whenPredict_thenExpectedPrediction() {
        var configuration = new NeuralNetworkConfiguration(
            new InputLayerConfiguration(
                2,
                List.of(1.0, 1.0)
            ),
            List.of(),
            new OutputLayerConfiguration(
                List.of(1.0)
            )
        );
        var neuralNetwork = NeuralNetwork.from(configuration);

        assertThat(neuralNetwork.prediction(new double[]{1, 1}))
            .isEqualTo(
                new double[]{
                    1 / (1 + Math.pow(Math.E, -(1 * 1 + 1 * 1 + 1)))
                }
            );
    }

    @Test
    void givenMultipleOutputs_whenPredict_thenExpectedPrediction() {
        var configuration = new NeuralNetworkConfiguration(
            new InputLayerConfiguration(
                1,
                List.of(1.0, 1.0)
            ),
            List.of(),
            new OutputLayerConfiguration(
                List.of(1.0, 1.0)
            )
        );
        var neuralNetwork = NeuralNetwork.from(configuration);

        var output = 1 / (1 + Math.pow(Math.E, -(1 * 1 + 1)));
        assertThat(neuralNetwork.prediction(new double[]{1}))
            .isEqualTo(new double[]{output, output});
    }
}
