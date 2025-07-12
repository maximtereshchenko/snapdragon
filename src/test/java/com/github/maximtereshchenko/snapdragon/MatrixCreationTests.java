package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class MatrixCreationTests {

    @Test
    void givenEmptyArray_whenCreateMatrix_thenIllegalArgumentExceptionThrown() {
        assertThatThrownBy(() -> Matrix.from(new double[0][0]))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenVariableLengthMatrix_whenCreateMatrix_thenIllegalArgumentExceptionThrown() {
        assertThatThrownBy(() -> Matrix.from(new double[][]{{1}, {2, 3}}))
            .isInstanceOf(IllegalArgumentException.class);
    }
}