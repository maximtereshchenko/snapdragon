package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

final class SigmoidTests {

    private final ActivationFunction activationFunction = new Sigmoid();

    @Test
    void givenSingleInput_whenApply_thenSingleActivatedOutput() {
        assertThat(activationFunction.apply(Tensor.horizontalVector(2)))
            .isEqualTo(Tensor.horizontalVector(1 / (1 + Math.exp(-2))));
    }

    @Test
    void givenMultipleInputs_whenApply_thenExpectedOutputs() {
        assertThat(activationFunction.apply(Tensor.matrix(2, 2, 2, 3, 4, 5)))
            .isEqualTo(
                Tensor.matrix(
                    2, 2,
                    1 / (1 + Math.exp(-2)), 1 / (1 + Math.exp(-3)),
                    1 / (1 + Math.exp(-4)), 1 / (1 + Math.exp(-5))
                )
            );
    }

    @Test
    void givenSingleInput_whenDeltas_thenExpectedDeltasValue() {
        assertThat(
            activationFunction.deltas(
                Tensor.horizontalVector(2),
                Tensor.horizontalVector(3)
            )
        )
            .isEqualTo(Tensor.horizontalVector(2 * (1 - 2) * 3));
    }

    @Test
    void givenMultipleInputs_whenDeltas_thenExpectedDeltas() {
        assertThat(
            activationFunction.deltas(
                Tensor.matrix(2, 2, 2, 3, 4, 5),
                Tensor.horizontalVector(6).broadcasted(2, 2)
            )
        )
            .isEqualTo(
                Tensor.matrix(
                    2, 2,
                    2 * (1 - 2) * 6, 3 * (1 - 3) * 6,
                    4 * (1 - 4) * 6, 5 * (1 - 5) * 6
                )
            );
    }
}