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
            () -> Tensor.matrix(0, 0),
            () -> Tensor.from(new int[]{1, 2, 0}, 1, 2, 3),
            () -> Tensor.matrix(2, 2, 1, 2, 3),
            () -> Tensor.horizontalVector(Double.NaN),
            () -> Tensor.verticalVector(Double.POSITIVE_INFINITY),
            () -> Tensor.matrix(1, 1, Double.NEGATIVE_INFINITY),
            () -> Tensor.from(new int[]{1, 1}, index -> Double.NaN)
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