package com.github.maximtereshchenko.snapdragon.matrix;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class MatrixTests {

    @Test
    void givenEmptyArray_whenCreateMatrix_thenIllegalArgumentExceptionThrown() {
        assertThatThrownBy(() -> Matrix.from(new double[0][0]))
            .isInstanceOf(IllegalArgumentException.class);
    }
}