package com.github.maximtereshchenko.snapdragon;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.DoubleBinaryOperator;
import java.util.function.ToDoubleFunction;

/**
 * Benchmark                                      Mode  Cnt           Score           Error  Units
 * ReferenceBasedTensorBenchmark.batchContracted  avgt    5   339791938.333 ±   2821413.067  ns/op
 * ReferenceBasedTensorBenchmark.broadcasted      avgt    5  3817482638.867 ± 100908619.399  ns/op
 * ReferenceBasedTensorBenchmark.contracted       avgt    5  3086805756.200 ±  77903664.535  ns/op
 * ReferenceBasedTensorBenchmark.lastValue        avgt    5         131.653 ±         3.428  ns/op
 * ReferenceBasedTensorBenchmark.sum              avgt    5   556131382.574 ±   5112955.680  ns/op
 * ReferenceBasedTensorBenchmark.transposed       avgt    5   440744669.930 ±   4785493.123  ns/op
 * <p>
 * Benchmark                                  Mode  Cnt         Score          Error  Units
 * ArrayBasedTensorBenchmark.batchContracted  avgt    5   2667709.475 ±    48775.030  ns/op
 * ArrayBasedTensorBenchmark.broadcasted      avgt    5         9.109 ±        0.148  ns/op
 * ArrayBasedTensorBenchmark.contracted       avgt    5   7653863.193 ±   229178.553  ns/op
 * ArrayBasedTensorBenchmark.lastValue        avgt    5         3.974 ±        0.059  ns/op
 * ArrayBasedTensorBenchmark.sum              avgt    5  98662410.208 ± 24821094.719  ns/op
 * ArrayBasedTensorBenchmark.transposed       avgt    5         2.683 ±        0.044  ns/op
 */
public final class Tensor {

    private final double[] values;
    private final Shape shape;

    private Tensor(double[] values, Shape shape) {
        this.values = values;
        this.shape = shape;
    }

    public static Tensor from(int[] shape, double... values) {
        var arrayBasedShape = ArrayBasedShape.from(shape);
        if (length(arrayBasedShape) != values.length) {
            throw new IllegalArgumentException();
        }
        var tensor = empty(arrayBasedShape);
        for (var i = 0; i < values.length; i++) {
            tensor.set(i, values[i]);
        }
        return tensor;
    }

    static Tensor from(int[] shape, ToDoubleFunction<int[]> function) {
        return from(ArrayBasedShape.from(shape), function);
    }

    static Tensor horizontalVector(double... values) {
        return matrix(1, values.length, values);
    }

    static Tensor verticalVector(double... values) {
        return matrix(values.length, 1, values);
    }

    static Tensor matrix(int rows, int columns, double... values) {
        return from(new int[]{rows, columns}, values);
    }

    private static Tensor from(Shape shape, ToDoubleFunction<int[]> function) {
        var tensor = empty(shape);
        try (var executor = Executors.newWorkStealingPool()) {
            fillAsync(executor, tensor, new IndexIterator(shape.array()), function);
        }
        return tensor;
    }

    private static void fillAsync(
        ExecutorService executor,
        Tensor tensor,
        Iterator<int[]> indexes,
        ToDoubleFunction<int[]> function
    ) {
        var futures = new ArrayList<CompletableFuture<Void>>();
        while (indexes.hasNext()) {
            var index = indexes.next();
            futures.add(
                CompletableFuture.runAsync(
                    () -> tensor.set(index, function.applyAsDouble(index)),
                    executor
                )
            );
        }
        await(futures);
    }

    private static void await(List<CompletableFuture<Void>> futures) {
        try {
            CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).join();
        } catch (CompletionException e) {
            if (e.getCause() instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
        }
    }

    private static Tensor empty(Shape shape) {
        return new Tensor(new double[length(shape)], shape);
    }

    private static int length(Shape shape) {
        var length = 1;
        for (var component : shape.array()) {
            length *= component;
        }
        return length;
    }

    @Override
    public int hashCode() {
        var result = 1.0;
        var iterator = new IndexIterator(shape.array());
        while (iterator.hasNext()) {
            result = 31 * result + value(iterator.next());
        }
        return (int) result;
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        if (!(object instanceof Tensor tensor && hasEqualShape(tensor))) {
            return false;
        }
        var iterator = new IndexIterator(shape.array());
        while (iterator.hasNext()) {
            var index = iterator.next();
            if (value(index) != tensor.value(index)) {
                return false;
            }
        }
        return true;
    }

    double value(int... index) {
        return values[shape.offset(index)];
    }

    int[] shape() {
        return shape.array();
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

    Tensor broadcasted(int... shape) {
        return new Tensor(
            values,
            BroadcastedShape.from(this.shape, ArrayBasedShape.from(shape))
        );
    }

    Tensor transposed() {
        return new Tensor(values, new TransposedShape(shape));
    }

    Tensor contracted(Tensor tensor) {
        return from(
            contracted(tensor.shape, 0),
            index -> productSum(tensor, index, 0)
        );
    }

    Tensor batchContracted(Tensor tensor) {
        return from(
            contracted(tensor.shape, 1),
            index -> productSum(tensor, index, 1)
        );
    }

    private Shape contracted(Shape shape, int dimension) {
        var thisShapeArray = this.shape.array();
        var otherShapeArray = shape.array();
        for (var i = 0; i < dimension; i++) {
            if (thisShapeArray[i] != otherShapeArray[i]) {
                throw new IllegalArgumentException();
            }
        }
        if (thisShapeArray[thisShapeArray.length - 1] != otherShapeArray[dimension]) {
            throw new IllegalArgumentException();
        }
        var contracted = new int[thisShapeArray.length + otherShapeArray.length - dimension - 2];
        System.arraycopy(
            thisShapeArray,
            0,
            contracted,
            0,
            thisShapeArray.length - 1
        );
        System.arraycopy(
            otherShapeArray,
            dimension + 1,
            contracted,
            thisShapeArray.length - 1,
            otherShapeArray.length - dimension - 1
        );
        return ArrayBasedShape.from(contracted);
    }

    private void set(int[] index, double value) {
        set(shape.offset(index), value);
    }

    private void set(int index, double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException();
        }
        values[index] = value;
    }

    private double productSum(Tensor tensor, int[] index, int dimension) {
        var thisShapeArray = shape.array();
        var left = new int[thisShapeArray.length];
        System.arraycopy(index, 0, left, 0, left.length - 1);
        var otherShapeArray = tensor.shape.array();
        var right = new int[otherShapeArray.length];
        System.arraycopy(index, 0, right, 0, dimension);
        System.arraycopy(
            index,
            thisShapeArray.length - 1,
            right,
            dimension + 1,
            right.length - dimension - 1
        );
        var sum = 0.0;
        for (
            var contractedDimension = 0;
            contractedDimension < thisShapeArray[thisShapeArray.length - 1];
            contractedDimension++
        ) {
            left[left.length - 1] = contractedDimension;
            right[dimension] = contractedDimension;
            sum += value(left) * tensor.value(right);
        }
        return sum;
    }

    private Tensor combined(Tensor tensor, DoubleBinaryOperator operator) {
        if (!hasEqualShape(tensor)) {
            throw new IllegalArgumentException();
        }
        return Tensor.from(
            ArrayBasedShape.from(shape.array()),
            index -> operator.applyAsDouble(value(index), tensor.value(index))
        );
    }

    private boolean hasEqualShape(Tensor tensor) {
        return Arrays.equals(shape(), tensor.shape());
    }

    private interface Shape {

        int offset(int[] index);

        int[] array();
    }

    private static final class ArrayBasedShape implements Shape {

        private final int[] components;

        private ArrayBasedShape(int[] components) {
            this.components = components;
        }

        static Shape from(int[] components) {
            if (components.length < 2 || hasNonPositive(components)) {
                throw new IllegalArgumentException();
            }
            return new ArrayBasedShape(components);
        }

        private static boolean hasNonPositive(int[] components) {
            for (var component : components) {
                if (component < 1) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public int offset(int[] index) {
            var offset = 0;
            for (var i = 0; i < index.length; i++) {
                var added = index[i];
                for (var next = i + 1; next < components.length; next++) {
                    added *= components[next];
                }
                offset += added;
            }
            return offset;
        }

        @Override
        public int[] array() {
            return components;
        }
    }

    private static final class BroadcastedShape implements Shape {

        private final Shape original;
        private final Shape broadcasted;

        private BroadcastedShape(Shape original, Shape broadcasted) {
            this.original = original;
            this.broadcasted = broadcasted;
        }

        static Shape from(Shape original, Shape broadcasted) {
            var broadcastedArray = broadcasted.array();
            var originalArray = original.array();
            int difference = broadcastedArray.length - originalArray.length;
            if (difference < 0) {
                throw new IllegalArgumentException();
            }
            for (var i = broadcastedArray.length - 1; i >= 0; i--) {
                var originalIndex = i - difference;
                if (
                    originalIndex >= 0 &&
                        originalArray[originalIndex] != 1 &&
                        originalArray[originalIndex] != broadcastedArray[i]
                ) {
                    throw new IllegalArgumentException();
                }
            }
            return new BroadcastedShape(original, broadcasted);
        }

        @Override
        public int offset(int[] index) {
            var originalArray = original.array();
            var originalIndex = new int[originalArray.length];
            var broadcastedArray = broadcasted.array();
            int difference = broadcastedArray.length - originalArray.length;
            for (var i = 0; i < originalIndex.length; i++) {
                originalIndex[i] = Math.min(originalArray[i] - 1, index[i + difference]);
            }
            return original.offset(originalIndex);
        }

        @Override
        public int[] array() {
            return broadcasted.array();
        }
    }

    private static final class TransposedShape implements Shape {

        private final Shape original;

        TransposedShape(Shape original) {
            this.original = original;
        }

        @Override
        public int offset(int[] index) {
            var reversedIndex = new int[index.length];
            System.arraycopy(index, 1, reversedIndex, 0, index.length - 1);
            reversedIndex[reversedIndex.length - 1] = index[0];
            return original.offset(reversedIndex);
        }

        @Override
        public int[] array() {
            var originalArray = original.array();
            var transposed = new int[originalArray.length];
            System.arraycopy(
                originalArray,
                0,
                transposed,
                1,
                originalArray.length - 1
            );
            transposed[0] = originalArray[originalArray.length - 1];
            return transposed;
        }
    }

    private static final class IndexIterator implements Iterator<int[]> {

        private final int[] index;
        private final int[] shape;
        private boolean advance = false;
        private boolean exhausted = false;

        IndexIterator(int[] shape) {
            this.index = new int[shape.length];
            this.shape = shape;
        }

        @Override
        public boolean hasNext() {
            advance();
            return !exhausted;
        }

        @Override
        public int[] next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            advance = true;
            var copy = new int[index.length];
            System.arraycopy(index, 0, copy, 0, index.length);
            return copy;
        }

        private void advance() {
            if (!advance || exhausted) {
                return;
            }
            var offset = index.length - 1;
            while (offset >= 0) {
                if (index[offset] == shape[offset] - 1) {
                    index[offset] = 0;
                    offset--;
                } else {
                    index[offset]++;
                    advance = false;
                    return;
                }
            }
            exhausted = true;
        }
    }
}
