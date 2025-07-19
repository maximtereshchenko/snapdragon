package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class TensorCombinationTests {

    private static Stream<Arguments> combinedTensors() {
        var inputs = List.of(
            Tensor.horizontalVector(2),
            Tensor.horizontalVector(1),
            Tensor.horizontalVector(4, 3),
            Tensor.horizontalVector(2, 1),
            Tensor.verticalVector(4, 3),
            Tensor.verticalVector(2, 1),
            Tensor.from(List.of(2, 2), 8, 7, 6, 5),
            Tensor.from(List.of(2, 2), 4, 3, 2, 1),
            Tensor.from(List.of(2, 2, 2), 16, 15, 14, 13, 12, 11, 10, 9),
            Tensor.from(List.of(2, 2, 2), 8, 7, 6, 5, 4, 3, 2, 1)
        );
        return Stream.of(
                arguments(
                    inputs,
                    Tensor::sum,
                    List.of(
                        Tensor.horizontalVector(2 + 1),
                        Tensor.horizontalVector(4 + 2, 3 + 1),
                        Tensor.verticalVector(4 + 2, 3 + 1),
                        Tensor.from(List.of(2, 2), 8 + 4, 7 + 3, 6 + 2, 5 + 1),
                        Tensor.from(
                            List.of(2, 2, 2),
                            16 + 8, 15 + 7, 14 + 6, 13 + 5, 12 + 4, 11 + 3, 10 + 2, 9 + 1
                        )
                    )
                ),
                arguments(
                    inputs,
                    Tensor::difference,
                    List.of(
                        Tensor.horizontalVector(2 - 1),
                        Tensor.horizontalVector(4 - 2, 3 - 1),
                        Tensor.verticalVector(4 - 2, 3 - 1),
                        Tensor.from(List.of(2, 2), 8 - 4, 7 - 3, 6 - 2, 5 - 1),
                        Tensor.from(
                            List.of(2, 2, 2),
                            16 - 8, 15 - 7, 14 - 6, 13 - 5, 12 - 4, 11 - 3, 10 - 2, 9 - 1
                        )
                    )
                ),
                arguments(
                    inputs,
                    Tensor::product,
                    List.of(
                        Tensor.horizontalVector(2 * 1),
                        Tensor.horizontalVector(4 * 2, 3 * 1),
                        Tensor.verticalVector(4 * 2, 3 * 1),
                        Tensor.from(List.of(2, 2), 8 * 4, 7 * 3, 6 * 2, 5 * 1),
                        Tensor.from(
                            List.of(2, 2, 2),
                            16 * 8, 15 * 7, 14 * 6, 13 * 5, 12 * 4, 11 * 3, 10 * 2, 9 * 1
                        )
                    )
                ),
                arguments(
                    inputs,
                    Tensor::quotient,
                    List.of(
                        Tensor.horizontalVector(2.0 / 1),
                        Tensor.horizontalVector(4.0 / 2, 3.0 / 1),
                        Tensor.verticalVector(4.0 / 2, 3.0 / 1),
                        Tensor.from(List.of(2, 2), 8.0 / 4, 7.0 / 3, 6.0 / 2, 5.0 / 1),
                        Tensor.from(
                            List.of(2, 2, 2),
                            16.0 / 8,
                            15.0 / 7,
                            14.0 / 6,
                            13.0 / 5,
                            12.0 / 4,
                            11.0 / 3,
                            10.0 / 2,
                            9.0 / 1
                        )
                    )
                )
            )
                   .flatMap(Collection::stream);
    }

    private static List<BinaryOperator<Tensor>> operators() {
        return List.of(Tensor::sum, Tensor::difference, Tensor::product, Tensor::quotient);
    }

    private static List<Arguments> arguments(
        List<Tensor> inputs,
        BinaryOperator<Tensor> operator,
        List<Tensor> expected
    ) {
        var arguments = new ArrayList<Arguments>();
        for (var i = 0; i < expected.size(); i++) {
            arguments.add(
                arguments(
                    inputs.get(i * 2),
                    inputs.get(i * 2 + 1),
                    operator,
                    expected.get(i)
                )
            );
        }
        return arguments;
    }

    private static Arguments arguments(
        Tensor first,
        Tensor second,
        BinaryOperator<Tensor> operator,
        Tensor expected
    ) {
        return Arguments.arguments(first, second, operator, expected);
    }

    @ParameterizedTest
    @MethodSource("operators")
    void givenDifferentHeight_whenCombined_thenIllegalArgumentExceptionThrown(
        BinaryOperator<Tensor> operator
    ) {
        var oneRow = Tensor.horizontalVector(1);
        var twoRows = Tensor.verticalVector(1, 2);
        assertThatThrownBy(() -> operator.apply(oneRow, twoRows))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @MethodSource("operators")
    void givenDifferentWidth_whenCombined_thenIllegalArgumentExceptionThrown(
        BinaryOperator<Tensor> operator
    ) {
        var oneColumn = Tensor.verticalVector(1);
        var twoColumns = Tensor.horizontalVector(1, 2);
        assertThatThrownBy(() -> operator.apply(oneColumn, twoColumns))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @MethodSource("combinedTensors")
    void givenTensors_whenCombined_thenExpectedTensor(
        Tensor first,
        Tensor second,
        BinaryOperator<Tensor> operator,
        Tensor expected
    ) {
        assertThat(operator.apply(first, second)).isEqualTo(expected);
    }
}