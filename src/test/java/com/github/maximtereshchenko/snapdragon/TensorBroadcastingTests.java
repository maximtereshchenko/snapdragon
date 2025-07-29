package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.params.provider.Arguments.arguments;

final class TensorBroadcastingTests {

    private static List<Arguments> broadcastedTensors() {
        return List.of(
            arguments(
                Tensor.verticalVector(1),
                new int[]{1, 1},
                Tensor.horizontalVector(1)
            ),
            arguments(
                Tensor.horizontalVector(1),
                new int[]{3, 3},
                Tensor.matrix(3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)
            ),
            arguments(
                Tensor.horizontalVector(1),
                new int[]{1, 3},
                Tensor.horizontalVector(1, 1, 1)
            ),
            arguments(
                Tensor.verticalVector(1),
                new int[]{3, 1},
                Tensor.verticalVector(1, 1, 1)
            ),
            arguments(
                Tensor.matrix(2, 2, 1, 2, 3, 4),
                new int[]{3, 2, 2},
                Tensor.from(
                    new int[]{3, 2, 2},
                    1, 2, 3, 4,
                    1, 2, 3, 4,
                    1, 2, 3, 4
                )
            ),
            arguments(
                Tensor.from(new int[]{3, 1, 2}, 1, 2, 3, 4, 5, 6),
                new int[]{3, 3, 2},
                Tensor.from(
                    new int[]{3, 3, 2},
                    1, 2,
                    1, 2,
                    1, 2,
                    3, 4,
                    3, 4,
                    3, 4,
                    5, 6,
                    5, 6,
                    5, 6
                )
            )
        );
    }

    private static List<Arguments> incompatibleTensors() {
        var horizontalVector = Tensor.horizontalVector(1, 2);
        var verticalVector = Tensor.verticalVector(1, 2);
        var matrix = Tensor.matrix(2, 2, 1, 2, 3, 4);
        var tensor = Tensor.from(new int[]{2, 2, 2}, 1, 2, 3, 4, 5, 6, 7, 8);
        return List.of(
            arguments(horizontalVector, new int[]{1, 3}),
            arguments(horizontalVector, new int[]{1, 1}),
            arguments(verticalVector, new int[]{3, 1}),
            arguments(verticalVector, new int[]{1, 1}),
            arguments(matrix, new int[]{1}),
            arguments(matrix, new int[]{1, 2}),
            arguments(matrix, new int[]{2, 1}),
            arguments(matrix, new int[]{1, 1}),
            arguments(matrix, new int[]{3, 3}),
            arguments(tensor, new int[]{1}),
            arguments(tensor, new int[]{1, 1}),
            arguments(tensor, new int[]{1, 2}),
            arguments(tensor, new int[]{2, 1}),
            arguments(tensor, new int[]{2, 2}),
            arguments(tensor, new int[]{2, 3}),
            arguments(tensor, new int[]{3, 2}),
            arguments(tensor, new int[]{3, 3}),
            arguments(tensor, new int[]{1, 1, 1}),
            arguments(tensor, new int[]{1, 1, 2}),
            arguments(tensor, new int[]{1, 2, 1}),
            arguments(tensor, new int[]{2, 1, 1}),
            arguments(tensor, new int[]{2, 2, 1}),
            arguments(tensor, new int[]{2, 1, 2}),
            arguments(tensor, new int[]{1, 2, 2}),
            arguments(tensor, new int[]{2, 2, 3}),
            arguments(tensor, new int[]{2, 3, 2}),
            arguments(tensor, new int[]{3, 2,})
        );
    }

    @ParameterizedTest
    @MethodSource("broadcastedTensors")
    void givenCompatibleShape_whenBroadcasted_thenBroadcastedTensor(
        Tensor tensor,
        int[] shape,
        Tensor broadcasted
    ) {
        assertThat(tensor.broadcasted(shape)).isEqualTo(broadcasted);
    }

    @ParameterizedTest
    @MethodSource("incompatibleTensors")
    void givenIncompatibleShape_whenBroadcasted_thenBroadcastedTensor(
        Tensor tensor,
        int[] shape
    ) {
        assertThatThrownBy(() -> tensor.broadcasted(shape))
            .isInstanceOf(IllegalArgumentException.class);
    }
}