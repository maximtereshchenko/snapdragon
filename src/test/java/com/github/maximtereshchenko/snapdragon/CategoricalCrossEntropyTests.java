package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class CategoricalCrossEntropyTests {

    private final LossFunction lossFunction = new CategoricalCrossEntropy();

    @Test
    void givenTwoClasses_whenLoss_thenExpectedLoss() {
        assertThat(
            lossFunction.loss(
                Tensor.horizontalVector(0.2, 0.8),
                Tensor.horizontalVector(0, 1)
            )
        )
            .isEqualTo(Tensor.horizontalVector(-(0 * Math.log(0.2) + 1 * Math.log(0.8))));
    }

    @Test
    void givenThreeClasses_whenLoss_thenExpectedLoss() {
        assertThat(
            lossFunction.loss(
                Tensor.horizontalVector(0.3, 0.6, 0.1),
                Tensor.horizontalVector(1, 0, 0)
            )
        )
            .isEqualTo(
                Tensor.horizontalVector(
                    -(1 * Math.log(0.3) + 0 * Math.log(0.6) + 0 * Math.log(0.1))
                )
            );
    }

    @Test
    void givenMultipleBatches_whenLoss_thenExpectedLoss() {
        assertThat(
            lossFunction.loss(
                Tensor.from(List.of(2, 2), 0.2, 0.8, 0.3, 0.7),
                Tensor.from(List.of(2, 2), 0, 1, 1, 0)
            )
        )
            .isEqualTo(
                Tensor.verticalVector(
                    -(0 * Math.log(0.2) + 1 * Math.log(0.8)),
                    -(1 * Math.log(0.3) + 0 * Math.log(0.7))
                )
            );
    }

    @Test
    void givenSingleOutputsRow_whenDerivative_thenExpectedDerivatives() {
        assertThat(
            lossFunction.derivative(
                Tensor.horizontalVector(0.2, 0.3, 0.5),
                Tensor.horizontalVector(0, 0, 1)
            )
        )
            .isEqualTo(
                Tensor.horizontalVector(-0.0, -0.0, -1 / 0.5)
            );
    }

    @Test
    void givenMultipleOutputRows_whenDerivative_thenExpectedDerivatives() {
        assertThat(
            lossFunction.derivative(
                Tensor.from(
                    List.of(2, 3),
                    0.1, 0.5, 0.4,
                    0.4, 0.3, 0.3
                ),
                Tensor.from(
                    List.of(2, 3),
                    0, 1, 0,
                    1, 0, 0
                )
            )
        )
            .isEqualTo(
                Tensor.from(
                    List.of(2, 3),
                    -0.0, -1 / 0.5, -0.0,
                    -1 / 0.4, -0.0, -0.0
                )
            );
    }
}