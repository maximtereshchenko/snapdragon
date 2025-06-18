package com.github.maximtereshchenko.snapdragon.neuronnetwork.domain;

import java.util.List;

interface ConnectableSegment {

    List<Connection> connections(int destinationIndex);
}
