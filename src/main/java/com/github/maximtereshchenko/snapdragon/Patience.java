package com.github.maximtereshchenko.snapdragon;

final class Patience {

    private final double bestLoss;
    private final int noImprovementEpochs;
    private final int maxNoImprovementEpochs;

    private Patience(double bestLoss, int noImprovementEpochs, int maxNoImprovementEpochs) {
        this.bestLoss = bestLoss;
        this.noImprovementEpochs = noImprovementEpochs;
        this.maxNoImprovementEpochs = maxNoImprovementEpochs;
    }

    Patience(int maxNoImprovementEpochs) {
        this(Double.MAX_VALUE, 0, maxNoImprovementEpochs);
    }

    Decision next(double loss) {
        if (loss < bestLoss) {
            return new Improvement(new Patience(loss, 0, maxNoImprovementEpochs));
        }
        if (noImprovementEpochs < maxNoImprovementEpochs) {
            return new NoImprovement(
                new Patience(bestLoss, noImprovementEpochs + 1, maxNoImprovementEpochs)
            );
        }
        return new Stop();
    }
}
