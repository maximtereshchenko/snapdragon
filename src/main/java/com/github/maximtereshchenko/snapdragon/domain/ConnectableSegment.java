package com.github.maximtereshchenko.snapdragon.domain;

import java.util.List;

interface ConnectableSegment {

    List<Connection> connections(int destinationIndex);
}
