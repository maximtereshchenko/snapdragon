package com.github.maximtereshchenko.snapdragon;

import java.util.*;
import java.util.function.DoubleBinaryOperator;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

final class Tensor {

    private final List<Tree> trees;

    private Tensor(List<Tree> trees) {
        this.trees = trees;
    }

    static Tensor from(List<Integer> shape, double... values) {
        var targetShape = new Shape(Index.from(shape));
        return from(
            targetShape,
            index -> {
                var arrayIndex = offset(targetShape, index);
                if (arrayIndex >= values.length) {
                    throw new IllegalArgumentException();
                }
                return values[arrayIndex];
            }
        );
    }

    static Tensor horizontalVector(double... values) {
        return from(List.of(1, values.length), values);
    }

    static Tensor verticalVector(double... values) {
        return from(List.of(values.length, 1), values);
    }

    static Tensor from(List<Integer> shape, ToDoubleFunction<List<Integer>> function) {
        return from(
            new Shape(Index.from(shape)),
            index -> function.applyAsDouble(index.components())
        );
    }

    private static int offset(Shape shape, Index index) {
        var offset = 0;
        for (var current = 0; current < index.size(); current++) {
            var added = index.value(current);
            for (var next = current + 1; next < shape.size(); next++) {
                added *= shape.value(next);
            }
            offset += added;
        }
        return offset;
    }

    private static Tensor from(Shape shape, ToDoubleFunction<Index> function) {
        if (shape.first() == 0) {
            throw new IllegalArgumentException();
        }
        return new Tensor(
            IntStream.range(0, shape.first())
                .mapToObj(Index::from)
                .map(prefix ->
                         tree(
                             shape.decapitated(),
                             index -> function.applyAsDouble(prefix.appended(index))
                         )
                )
                .toList()
        );
    }

    private static Tree tree(Shape shape, ToDoubleFunction<Index> function) {
        if (shape.size() == 0) {
            return new Leaf(function.applyAsDouble(Index.from()));
        }
        return new Branch(from(shape, function));
    }

    @Override
    public int hashCode() {
        return Objects.hash(trees);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof Tensor tensor &&
                   Objects.equals(trees, tensor.trees);
    }

    double value(Integer... index) {
        return value(List.of(index));
    }

    double value(List<Integer> index) {
        return value(Index.from(index));
    }

    List<Integer> shape() {
        var shape = new ArrayList<Integer>();
        for (
            Tree current = new Branch(this);
            current instanceof Branch(var tensor);
            current = tensor.trees.getFirst()
        ) {
            shape.add(tensor.trees.size());
        }
        return shape;
    }

    Tensor sum(Tensor tensor) {
        return combined(tensor, Double::sum);
    }

    Tensor difference(Tensor tensor) {
        return combined(tensor, (first, second) -> first - second);
    }

    Tensor product(Tensor tensor) {
        return combined(tensor, (first, second) -> first * second);
    }

    Tensor quotient(Tensor tensor) {
        return combined(tensor, (first, second) -> first / second);
    }

    Tensor broadcasted(Integer... shape) {
        return broadcasted(List.of(shape));
    }

    Tensor broadcasted(List<Integer> shape) {
        var shapeInstance = shapeInstance();
        return Tensor.from(
            shapeInstance.broadcasted(shape),
            index -> value(shapeInstance.indexFromBroadcasted(index))
        );
    }

    Tensor transposed() {
        return Tensor.from(shapeInstance().transposed(), index -> value(index.transposed()));
    }

    Tensor contracted(Tensor tensor) {
        var shape = shapeInstance();
        var contracted = shape.contracted(tensor.shapeInstance());
        return Tensor.from(
            contracted,
            index -> IntStream.range(0, shape.last())
                         .mapToObj(Index::from)
                         .mapToDouble(position -> product(shape, tensor, index, position))
                         .sum()
        );
    }

    Tensor batchContracted(Tensor tensor) {
        var shape = shapeInstance();
        if (!shape.isBatchContractable(tensor.shapeInstance())) {
            throw new IllegalArgumentException();
        }
        var contractedTrees = new ArrayList<Tree>();
        for (var i = 0; i < shape.first(); i++) {
            var index = Index.from(i);
            if (tree(index) instanceof Branch(var first) &&
                    tensor.tree(index) instanceof Branch(var second)) {
                contractedTrees.add(new Branch(first.contracted(second)));
            }
        }
        return new Tensor(contractedTrees);
    }

    private double product(Shape shape, Tensor tensor, Index index, Index position) {
        return value(index.slice(0, shape.size() - 1).appended(position)) *
                   tensor.value(position.appended(index.slice(shape.size() - 1, index.size())));
    }

    private Tensor combined(Tensor tensor, DoubleBinaryOperator operator) {
        var shape = shapeInstance();
        if (!shape.equals(tensor.shapeInstance())) {
            throw new IllegalArgumentException();
        }
        return Tensor.from(
            shape,
            index -> operator.applyAsDouble(value(index), tensor.value(index))
        );
    }

    private double value(Index index) {
        if (!(tree(index) instanceof Leaf(var value))) {
            throw new IllegalArgumentException();
        }
        return value;
    }

    private Tree tree(Index index) {
        var tree = trees.get(index.first());
        if (index.size() == 1) {
            return tree;
        }
        if (!(tree instanceof Branch(var tensor))) {
            throw new IllegalArgumentException();
        }
        return tensor.tree(index.decapitated());
    }

    private Shape shapeInstance() {
        return new Shape(Index.from(shape()));
    }

    private sealed interface Tree {}

    private record Branch(Tensor tensor) implements Tree {}

    private record Leaf(double value) implements Tree {}

    private static final class Shape {

        private final Index max;

        Shape(Index max) {
            this.max = max;
        }

        @Override
        public int hashCode() {
            return Objects.hash(max);
        }

        @Override
        public boolean equals(Object object) {
            if (this == object) {
                return true;
            }
            return object instanceof Shape shape &&
                       Objects.equals(max, shape.max);
        }

        int first() {
            return max.first();
        }

        int last() {
            return max.last();
        }

        int value(int offset) {
            return max.value(offset);
        }

        int size() {
            return max.size();
        }

        Shape broadcasted(List<Integer> shape) {
            var broadcasted = Index.from(shape);
            var padded = max.padded(shape.size());
            for (var i = 0; i < broadcasted.size(); i++) {
                var value = padded.value(i);
                if (value != 1 && value != broadcasted.value(i)) {
                    throw new IllegalArgumentException();
                }
            }
            return new Shape(broadcasted);
        }

        Shape contracted(Shape shape) {
            return new Shape(max.contracted(shape.max).orElseThrow(IllegalArgumentException::new));
        }

        boolean isBatchContractable(Shape shape) {
            if (first() != shape.first()) {
                return false;
            }
            return max.decapitated().contracted(shape.max.decapitated()).isPresent();
        }

        Shape transposed() {
            return new Shape(max.transposed());
        }

        Index indexFromBroadcasted(Index broadcasted) {
            var index = new ArrayList<Integer>();
            var difference = broadcasted.size() - size();
            for (var offset = size() - 1; offset >= 0; offset--) {
                index.addFirst(
                    Math.min(max.value(offset) - 1, broadcasted.value(offset + difference))
                );
            }
            return Index.from(index);
        }

        Shape decapitated() {
            return new Shape(max.decapitated());
        }
    }

    private static final class Index {

        private final List<Integer> components;

        private Index(List<Integer> components) {
            this.components = components;
        }

        static Index from(int... components) {
            return new Index(Arrays.stream(components).boxed().toList());
        }

        static Index from(List<Integer> components) {
            if (hasNegative(components)) {
                throw new IllegalArgumentException();
            }
            return new Index(components);
        }

        private static boolean hasNegative(List<Integer> components) {
            return components.stream().anyMatch(component -> component < 0);
        }

        @Override
        public int hashCode() {
            return Objects.hash(components);
        }

        @Override
        public boolean equals(Object object) {
            if (this == object) {
                return true;
            }
            return object instanceof Index index &&
                       Objects.equals(components, index.components);
        }

        List<Integer> components() {
            return components;
        }

        int first() {
            return components.getFirst();
        }

        int value(int offset) {
            return components.get(offset);
        }

        int last() {
            return components.getLast();
        }

        int size() {
            return components.size();
        }

        Index padded(int size) {
            if (size < size()) {
                throw new IllegalArgumentException();
            }
            var padded = this;
            while (padded.size() != size) {
                padded = Index.from(1).appended(padded);
            }
            return padded;
        }

        Optional<Index> contracted(Index index) {
            if (last() != index.first() || size() < 1 || index.size() < 2) {
                return Optional.empty();
            }
            return Optional.of(slice(0, size() - 1).appended(index.slice(1, index.size())));
        }

        Index transposed() {
            return Index.from(components.getLast())
                       .appended(Index.from(components.subList(0, size() - 1)));
        }

        Index slice(int from, int to) {
            return Index.from(components.subList(from, to));
        }

        Index decapitated() {
            return slice(1, size());
        }

        Index appended(Index index) {
            var extended = new ArrayList<>(components);
            extended.addAll(index.components);
            return Index.from(extended);
        }
    }
}
