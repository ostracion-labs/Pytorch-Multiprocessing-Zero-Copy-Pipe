from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PacketDto:
    """Immutable metadata packet passed across multiprocess queues."""

    group_id: int
    group_index: int
    dimension: tuple[int, ...]
    data_id: int
