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
                List.of(1, 1),
                Tensor.horizontalVector(1)
            ),
            arguments(
                Tensor.horizontalVector(1),
                List.of(3, 3),
                Tensor.from(List.of(3, 3), 1, 1, 1, 1, 1, 1, 1, 1, 1)
            ),
            arguments(
                Tensor.horizontalVector(1),
                List.of(1, 3),
                Tensor.horizontalVector(1, 1, 1)
            ),
            arguments(
                Tensor.verticalVector(1),
                List.of(3, 1),
                Tensor.verticalVector(1, 1, 1)
            ),
            arguments(
                Tensor.from(List.of(2, 2), 1, 2, 3, 4),
                List.of(3, 2, 2),
                Tensor.from(
                    List.of(3, 2, 2),
                    1, 2, 3, 4,
                    1, 2, 3, 4,
                    1, 2, 3, 4
                )
            ),
            arguments(
                Tensor.from(List.of(3, 1, 2), 1, 2, 3, 4, 5, 6),
                List.of(3, 3, 2),
                Tensor.from(
                    List.of(3, 3, 2),
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
        var matrix = Tensor.from(List.of(2, 2), 1, 2, 3, 4);
        var tensor = Tensor.from(List.of(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8);
        return List.of(
            arguments(horizontalVector, List.of(1, 3)),
            arguments(horizontalVector, List.of(1, 1)),
            arguments(verticalVector, List.of(3, 1)),
            arguments(verticalVector, List.of(1, 1)),
            arguments(matrix, List.of(1)),
            arguments(matrix, List.of(1, 2)),
            arguments(matrix, List.of(2, 1)),
            arguments(matrix, List.of(1, 1)),
            arguments(matrix, List.of(3, 3)),
            arguments(tensor, List.of(1)),
            arguments(tensor, List.of(1, 1)),
            arguments(tensor, List.of(1, 2)),
            arguments(tensor, List.of(2, 1)),
            arguments(tensor, List.of(2, 2)),
            arguments(tensor, List.of(2, 3)),
            arguments(tensor, List.of(3, 2)),
            arguments(tensor, List.of(3, 3)),
            arguments(tensor, List.of(1, 1, 1)),
            arguments(tensor, List.of(1, 1, 2)),
            arguments(tensor, List.of(1, 2, 1)),
            arguments(tensor, List.of(2, 1, 1)),
            arguments(tensor, List.of(2, 2, 1)),
            arguments(tensor, List.of(2, 1, 2)),
            arguments(tensor, List.of(1, 2, 2)),
            arguments(tensor, List.of(2, 2, 3)),
            arguments(tensor, List.of(2, 3, 2)),
            arguments(tensor, List.of(3, 2, 2))
        );
    }

    @ParameterizedTest
    @MethodSource("broadcastedTensors")
    void givenCompatibleShape_whenBroadcasted_thenBroadcastedTensor(
        Tensor tensor,
        List<Integer> shape,
        Tensor broadcasted
    ) {
        assertThat(tensor.broadcasted(shape)).isEqualTo(broadcasted);
    }

    @ParameterizedTest
    @MethodSource("incompatibleTensors")
    void givenIncompatibleShape_whenBroadcasted_thenBroadcastedTensor(
        Tensor tensor,
        List<Integer> shape
    ) {
        assertThatThrownBy(() -> tensor.broadcasted(shape))
            .isInstanceOf(IllegalArgumentException.class);
    }
}