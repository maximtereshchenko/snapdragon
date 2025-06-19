package com.github.maximtereshchenko.snapdragon.neuronnetwork;

import com.github.maximtereshchenko.snapdragon.matrix.Matrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class NeuralNetworkTrainingTests extends BaseNeuralNetworkTest {

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

    private double categoricalCrossEntropyDerivative(double label, double output) {
        return -label / output;
    }

    private double sigmoidDerivative(double output) {
        return output * (1 - output);
    }
}
