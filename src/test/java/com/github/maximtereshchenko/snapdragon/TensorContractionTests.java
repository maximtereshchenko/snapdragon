package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.params.provider.Arguments.arguments;

final class TensorContractionTests {

    private static List<Arguments> contractedTensors() {
        return List.of(
            arguments(
                Tensor.horizontalVector(2),
                Tensor.horizontalVector(3),
                Tensor.horizontalVector(6)
            ),
            arguments(
                Tensor.verticalVector(2, 3),
                Tensor.horizontalVector(4),
                Tensor.verticalVector(2 * 4, 3 * 4)
            ),
            arguments(
                Tensor.horizontalVector(1, 2),
                Tensor.verticalVector(3, 4),
                Tensor.horizontalVector(1 * 3 + 2 * 4)
            ),
            arguments(
                Tensor.matrix(2, 2, 1, 2, 3, 4),
                Tensor.matrix(2, 2, 5, 6, 7, 8),
                Tensor.matrix(
                    2, 2,
                    1 * 5 + 2 * 7, 1 * 6 + 2 * 8,
                    3 * 5 + 4 * 7, 3 * 6 + 4 * 8
                )
            ),
            arguments(
                Tensor.from(
                    new int[]{2, 3, 4},
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                    1, 0, 0, 0,
                    0, 1, 0, 0
                ),
                Tensor.from(
                    new int[]{4, 2, 2},
                    10, 11,
                    12, 13,
                    20, 21,
                    22, 23,
                    30, 31,
                    32, 33,
                    40, 41,
                    42, 43
                ),
                Tensor.from(
                    new int[]{2, 3, 2, 2},
                    10, 11, 12, 13,
                    20, 21, 22, 23,
                    30, 31, 32, 33,
                    40, 41, 42, 43,
                    10, 11, 12, 13,
                    20, 21, 22, 23
                )
            )
        );
    }

    private static List<Arguments> incompatibleTensors() {
        return List.of(
            arguments(
                Tensor.horizontalVector(1, 2),
                Tensor.horizontalVector(1)
            ),
            arguments(
                Tensor.horizontalVector(1, 2, 3),
                Tensor.verticalVector(1, 2)
            ),
            arguments(
                Tensor.matrix(2, 2, 1, 2, 3, 4),
                Tensor.horizontalVector(1)
            ),
            arguments(
                Tensor.matrix(2, 2, 1, 2, 3, 4),
                Tensor.from(new int[]{3, 2, 2}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            )
        );
    }

    @ParameterizedTest
    @MethodSource("contractedTensors")
    void givenCompatibleTensors_whenContracted_thenContractedTensor(
        Tensor first,
        Tensor second,
        Tensor contracted
    ) {
        assertThat(first.contracted(second)).isEqualTo(contracted);
    }

    @ParameterizedTest
    @MethodSource("incompatibleTensors")
    void givenIncompatibleTensors_whenContracted_thenIllegalArgumentExceptionThrown(
        Tensor first,
        Tensor second
    ) {
        assertThatThrownBy(() -> first.contracted(second))
            .isInstanceOf(IllegalArgumentException.class);
    }
}