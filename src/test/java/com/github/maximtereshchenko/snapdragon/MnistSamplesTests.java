package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class MnistSamplesTests {

    @Test
    void givenMultipleDimensionSample_thenSingleEntryDataset(@TempDir Path path)
        throws IOException {
        var inputs = Files.write(path.resolve("inputs"), bytes(List.of(1, 2, 2), 3, 4, 5, 6));
        var labels = Files.write(path.resolve("labels"), bytes(List.of(1), 7));

        assertThat(batchedLabeledSamples(inputs, labels))
            .toIterable()
            .containsExactly(
                new BatchedLabeledSample(
                    new Inputs(
                        Tensor.horizontalVector(3.0 / 255, 4.0 / 255, 5.0 / 255, 6.0 / 255)
                    ),
                    new Labels(Tensor.horizontalVector(0, 0, 0, 0, 0, 0, 0, 1, 0, 0))
                )
            );
    }

    @Test
    void givenMultipleSamples_thenMultipleDatasetEntries(@TempDir Path path)
        throws IOException {
        var inputs = Files.write(
            path.resolve("inputs"),
            bytes(List.of(2, 2, 2), 3, 4, 5, 6, 7, 8, 9, 10)
        );
        var labels = Files.write(path.resolve("labels"), bytes(List.of(2), 0, 1));

        assertThat(batchedLabeledSamples(inputs, labels))
            .toIterable()
            .containsExactly(
                new BatchedLabeledSample(
                    new Inputs(
                        Tensor.matrix(
                            2, 4,
                            3.0 / 255, 4.0 / 255,
                            5.0 / 255, 6.0 / 255,
                            7.0 / 255, 8.0 / 255,
                            9.0 / 255, 10.0 / 255
                        )
                    ),
                    new Labels(
                        Tensor.matrix(
                            2, 10,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 1, 0, 0, 0, 0, 0, 0, 0, 0
                        )
                    )
                )
            );
    }

    private Iterator<BatchedLabeledSample> batchedLabeledSamples(Path inputs, Path labels) {
        return new MnistSamples<>(
            inputs,
            labels,
            labeledSamples -> new ValidationDataset(labeledSamples, Integer.MAX_VALUE)
        )
                   .dataset()
                   .batchedLabeledSamples();
    }

    private byte[] bytes(List<Integer> shape, int... data) {
        var byteBuffer = ByteBuffer.allocate(4 + shape.size() * 4 + data.length);
        byteBuffer.put((byte) 0);
        byteBuffer.put((byte) 0);
        byteBuffer.put((byte) 8);
        byteBuffer.put((byte) shape.size());
        shape.forEach(byteBuffer::putInt);
        for (var datum : data) {
            byteBuffer.put((byte) datum);
        }
        return byteBuffer.array();
    }
}