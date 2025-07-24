package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class SoftmaxTests {

    private final ActivationFunction activationFunction = new Softmax();

    @Test
    void givenSingleInputsRow_whenApply_thenExpectedOutputs() {
        var sum = Math.exp(1) + Math.exp(3) + Math.exp(5);
        assertThat(activationFunction.apply(Tensor.horizontalVector(1, 3, 5)))
            .isEqualTo(
                Tensor.horizontalVector(
                    Math.exp(1) / sum,
                    Math.exp(3) / sum,
                    Math.exp(5) / sum
                )
            );
    }

    @Test
    void givenMultipleInputsRows_whenApply_thenExpectedOutputs() {
        var firstSum = Math.exp(2) + Math.exp(3) + Math.exp(4);
        var secondSum = Math.exp(5) + Math.exp(6) + Math.exp(7);
        assertThat(
            activationFunction.apply(
                Tensor.from(
                    List.of(2, 3),
                    2, 3, 4,
                    5, 6, 7
                )
            )
        )
            .isEqualTo(
                Tensor.from(
                    List.of(2, 3),
                    Math.exp(2) / firstSum, Math.exp(3) / firstSum, Math.exp(4) / firstSum,
                    Math.exp(5) / secondSum, Math.exp(6) / secondSum, Math.exp(7) / secondSum
                )
            );
    }

    @Test
    void givenTwoClasses_whenDeltas_thenBatchContractedTensor() {
        assertThat(
            activationFunction.deltas(
                Tensor.horizontalVector(0.25, 0.75),
                Tensor.horizontalVector(20, 4)
            )
        )
            .isEqualTo(
                Tensor.horizontalVector(
                    20 * 0.25 * (1 - 0.25) + 4 * 0.25 * (0 - 0.75),
                    20 * 0.75 * (0 - 0.25) + 4 * 0.75 * (1 - 0.75)
                )
            );
    }

    @Test
    void givenThreeClasses_whenDeltas_thenBatchContractedTensor() {
        assertThat(
            activationFunction.deltas(
                Tensor.horizontalVector(0.25, 0.125, 0.625),
                Tensor.horizontalVector(10, 23, 1)
            )
        )
            .isEqualTo(
                Tensor.horizontalVector(
                    10 * 0.25 * (1 - 0.25) + 23 * 0.25 * (0 - 0.125) + 1 * 0.25 * (0 - 0.625),
                    10 * 0.125 * (0 - 0.25) + 23 * 0.125 * (1 - 0.125) + 1 * 0.125 * (0 - 0.625),
                    10 * 0.625 * (0 - 0.25) + 23 * 0.625 * (0 - 0.125) + 1 * 0.625 * (1 - 0.625)
                )
            );
    }

    @Test
    void givenMultipleBatches_whenDeltas_thenBatchContractedTensors() {
        assertThat(
            activationFunction.deltas(
                Tensor.from(List.of(2, 2), 0.125, 0.875, 0.75, 0.25),
                Tensor.from(List.of(2, 2), 70, 6, 35, 3)
            )
        )
            .isEqualTo(
                Tensor.from(
                    List.of(2, 2),
                    70 * 0.125 * (1 - 0.125) + 6 * 0.125 * (0 - 0.875),
                    70 * 0.875 * (0 - 0.125) + 6 * 0.875 * (1 - 0.875),
                    35 * 0.75 * (1 - 0.75) + 3 * 0.75 * (0 - 0.25),
                    35 * 0.25 * (0 - 0.75) + 3 * 0.25 * (1 - 0.25)
                )
            );
    }
}