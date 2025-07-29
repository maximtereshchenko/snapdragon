package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.arguments;

final class TensorTranspositionTests {

    private static List<Arguments> transposedTensors() {
        return List.of(
            arguments(
                Tensor.verticalVector(1),
                Tensor.horizontalVector(1)
            ),
            arguments(
                Tensor.verticalVector(1, 2),
                Tensor.horizontalVector(1, 2)
            ),
            arguments(
                Tensor.horizontalVector(1, 2),
                Tensor.verticalVector(1, 2)
            ),
            arguments(
                Tensor.horizontalVector(1, 2),
                Tensor.verticalVector(1, 2)
            ),
            arguments(
                Tensor.matrix(2, 2, 1, 2, 3, 4),
                Tensor.matrix(2, 2, 4, 1, 2, 3)
            ),
            arguments(
                Tensor.from(new int[]{3, 1, 2}, 1, 2, 3, 4, 5, 6),
                Tensor.from(new int[]{2, 3, 1}, 1, 2, 3, 4, 5, 6)
            )
        );
    }

    @ParameterizedTest
    @MethodSource("transposedTensors")
    void givenTensor_whenTransposed_thenTransposedTensor() {
        var matrix = Tensor.horizontalVector(1);
        assertThat(matrix.transposed()).isEqualTo(matrix);
    }
}