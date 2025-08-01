package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.assertThat;

final class ReLUTests {

    private final ActivationFunction activationFunction = new ReLU();

    @ParameterizedTest
    @CsvSource(
        textBlock = """
                    -1, 0
                    -0.5, 0
                    0, 0
                    0.5, 0.5
                    1, 1
                    """
    )
    void givenSingleInput_whenApply_thenSingleActivatedOutput(double input, double output) {
        assertThat(activationFunction.apply(Tensor.horizontalVector(input)))
            .isEqualTo(Tensor.horizontalVector(output));
    }

    @Test
    void givenMultipleInputs_whenApply_thenExpectedOutputs() {
        assertThat(
            activationFunction.apply(
                Tensor.matrix(
                    2, 3,
                    -1, -0.5, 0,
                    0, 0.5, 1
                )
            )
        )
            .isEqualTo(
                Tensor.matrix(
                    2, 3,
                    0, 0, 0,
                    0, 0.5, 1
                )
            );
    }

    @ParameterizedTest
    @CsvSource(
        textBlock = """
                    -1, 0
                    -0.5, 0
                    0, 0
                    0.5, 1
                    1, 1
                    """
    )
    void givenSingleInput_whenDeltas_thenExpectedDeltasValue(
        double input,
        double derivative
    ) {
        assertThat(
            activationFunction.deltas(
                Tensor.horizontalVector(input),
                Tensor.horizontalVector(2)
            )
        )
            .isEqualTo(Tensor.horizontalVector(derivative * 2));
    }

    @Test
    void givenMultipleInputs_whenDeltas_thenExpectedDeltas() {
        assertThat(
            activationFunction.deltas(
                Tensor.matrix(
                    2, 3,
                    -1, -0.5, 0,
                    0, 0.5, 1
                ),
                Tensor.horizontalVector(2).broadcasted(2, 3)
            )
        )
            .isEqualTo(
                Tensor.matrix(
                    2, 3,
                    0, 0, 0,
                    0, 2, 2
                )
            );
    }
}