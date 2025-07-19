package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.params.provider.Arguments.arguments;

final class TensorBatchContractionTests {

    private static Stream<Arguments> batchContractedTensors() {
        return Stream.of(
            arguments(
                Tensor.from(List.of(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8),
                Tensor.from(List.of(2, 2, 2), 9, 10, 11, 12, 13, 14, 15, 16),
                Tensor.from(
                    List.of(2, 2, 2),
                    1 * 9 + 2 * 11, 1 * 10 + 2 * 12,
                    3 * 9 + 4 * 11, 3 * 10 + 4 * 12,
                    5 * 13 + 6 * 15, 5 * 14 + 6 * 16,
                    7 * 13 + 8 * 15, 7 * 14 + 8 * 16
                )
            ),
            arguments(
                Tensor.from(List.of(2, 1, 2), 1, 2, 3, 4),
                Tensor.from(List.of(2, 2, 1), 5, 6, 7, 8),
                Tensor.from(
                    List.of(2, 1, 1),
                    1 * 5 + 2 * 6,
                    3 * 7 + 4 * 8
                )
            ),
            arguments(
                Tensor.from(List.of(2, 2, 1), 1, 2, 3, 4),
                Tensor.from(List.of(2, 1, 2), 5, 6, 7, 8),
                Tensor.from(
                    List.of(2, 2, 2),
                    1 * 5, 1 * 6,
                    2 * 5, 2 * 6,
                    3 * 7, 3 * 8,
                    4 * 7, 4 * 8
                )
            ),
            arguments(
                Tensor.from(
                    List.of(2, 2, 2, 3),
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,
                    10, 11, 12,
                    13, 14, 15,
                    16, 17, 18,
                    19, 20, 21,
                    22, 23, 24
                ),
                Tensor.from(
                    List.of(2, 3, 2, 2),
                    1, 0,
                    0, 1,
                    1, 1,
                    1, 1,
                    2, 2,
                    2, 2,
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10,
                    11, 12
                ),
                Tensor.from(
                    List.of(2, 2, 2, 2, 2),
                    1 * 1 + 2 * 1 + 3 * 2,
                    1 * 0 + 2 * 1 + 3 * 2,
                    1 * 0 + 2 * 1 + 3 * 2,
                    1 * 1 + 2 * 1 + 3 * 2,
                    4 * 1 + 5 * 1 + 6 * 2,
                    4 * 0 + 5 * 1 + 6 * 2,
                    4 * 0 + 5 * 1 + 6 * 2,
                    4 * 1 + 5 * 1 + 6 * 2,
                    7 * 1 + 8 * 1 + 9 * 2,
                    7 * 0 + 8 * 1 + 9 * 2,
                    7 * 0 + 8 * 1 + 9 * 2,
                    7 * 1 + 8 * 1 + 9 * 2,
                    10 * 1 + 11 * 1 + 12 * 2,
                    10 * 0 + 11 * 1 + 12 * 2,
                    10 * 0 + 11 * 1 + 12 * 2,
                    10 * 1 + 11 * 1 + 12 * 2,
                    13 * 1 + 14 * 5 + 15 * 9,
                    13 * 2 + 14 * 6 + 15 * 10,
                    13 * 3 + 14 * 7 + 15 * 11,
                    13 * 4 + 14 * 8 + 15 * 12,
                    16 * 1 + 17 * 5 + 18 * 9,
                    16 * 2 + 17 * 6 + 18 * 10,
                    16 * 3 + 17 * 7 + 18 * 11,
                    16 * 4 + 17 * 8 + 18 * 12,
                    19 * 1 + 20 * 5 + 21 * 9,
                    19 * 2 + 20 * 6 + 21 * 10,
                    19 * 3 + 20 * 7 + 21 * 11,
                    19 * 4 + 20 * 8 + 21 * 12,
                    22 * 1 + 23 * 5 + 24 * 9,
                    22 * 2 + 23 * 6 + 24 * 10,
                    22 * 3 + 23 * 7 + 24 * 11,
                    22 * 4 + 23 * 8 + 24 * 12
                )
            )
        );
    }

    private static Stream<Arguments> incompatibleBatchTensors() {
        return Stream.of(
            arguments(
                Tensor.from(List.of(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8),
                Tensor.from(List.of(3, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            ),
            arguments(
                Tensor.from(List.of(2, 2, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                Tensor.from(List.of(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8)
            ),
            arguments(
                Tensor.from(List.of(2, 2), 1, 2, 3, 4),
                Tensor.from(List.of(2, 2), 1, 2, 3, 4)
            ),
            arguments(
                Tensor.from(List.of(2, 2), 1, 2, 3, 4),
                Tensor.from(List.of(1, 2, 2), 1, 2, 3, 4)
            )
        );
    }

    @ParameterizedTest
    @MethodSource("batchContractedTensors")
    void givenCompatibleTensors_whenBatchContracted_thenExpectedTensor(
        Tensor first,
        Tensor second,
        Tensor expected
    ) {
        assertThat(first.batchContracted(second)).isEqualTo(expected);
    }

    @ParameterizedTest
    @MethodSource("incompatibleBatchTensors")
    void givenIncompatibleTensors_whenBatchContracted_thenIllegalArgumentExceptionThrown(
        Tensor first,
        Tensor second
    ) {
        assertThatThrownBy(() -> first.batchContracted(second))
            .isInstanceOf(IllegalArgumentException.class);
    }
}
