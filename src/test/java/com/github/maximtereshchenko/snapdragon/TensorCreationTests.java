package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class TensorCreationTests {

    private static List<Supplier<Tensor>> emptyTensors() {
        return List.of(
            Tensor::horizontalVector,
            Tensor::verticalVector,
            () -> Tensor.from(List.of(0, 0)),
            () -> Tensor.from(List.of(1, 2, 0), 1, 2, 3),
            () -> Tensor.from(List.of(2, 2), 1, 2, 3)
        );
    }

    @ParameterizedTest
    @MethodSource("emptyTensors")
    void givenNoValues_whenCreateTensor_thenIllegalArgumentExceptionThrown(
        Supplier<Tensor> supplier
    ) {
        assertThatThrownBy(supplier::get).isInstanceOf(IllegalArgumentException.class);
    }
}