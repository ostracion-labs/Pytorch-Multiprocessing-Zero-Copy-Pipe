import dataclasses
import pickle
import uuid

import numpy as np
import pytest

from zero_copy_pipe import ActiveGroupTracker, BufferSpec, PacketDto


def test_packet_dto_is_frozen_and_slotted() -> None:
    pkt = PacketDto(group_id=1, group_index=2, dimension=(480, 640, 3), data_id=0)
    assert not hasattr(pkt, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        pkt.data_id = 99  # type: ignore[misc]


def test_packet_dto_pickle_roundtrip() -> None:
    gid = uuid.uuid4().int
    pkt = PacketDto(group_id=gid, group_index=3, dimension=(1080, 1920), data_id=2)
    restored = pickle.loads(pickle.dumps(pkt))
    assert restored == pkt


def test_buffer_spec_properties_and_validation() -> None:
    spec = BufferSpec((10, 20, 3), "uint16")
    assert spec.shape == (10, 20, 3)
    assert spec.dtype_str == "uint16"
    assert spec.numpy_dtype == np.dtype(np.uint16)
    assert spec.numel == 600
    assert spec.nbytes == 1200

    with pytest.raises(ValueError):
        BufferSpec((), "uint8")
    with pytest.raises(ValueError):
        BufferSpec((10, 0), "uint8")
    with pytest.raises(ValueError):
        BufferSpec((10, -4), "uint8")
    with pytest.raises(ValueError):
        BufferSpec((10, 10), "not_a_dtype")
    with pytest.raises(ValueError):
        BufferSpec((10, 10), "object")


def test_active_group_tracker_128bit_and_skipped_count() -> None:
    tracker = ActiveGroupTracker()
    assert tracker.active_group_id is None
    assert tracker.is_stale(0) is False
    assert tracker.is_stale(uuid.uuid4().int) is False

    gid_a = uuid.uuid4().int
    gid_b = (1 << 128) - 1
    tracker.set_active_group(gid_a)
    assert tracker.active_group_id == gid_a
    assert tracker.is_stale(gid_a) is False
    assert tracker.is_stale(gid_b) is True

    tracker.set_active_group(gid_b)
    assert tracker.active_group_id == gid_b
    assert tracker.is_stale(gid_b) is False
    assert tracker.is_stale(gid_a) is True

    tracker.clear_active_group()
    assert tracker.active_group_id is None
    assert tracker.is_stale(gid_a) is False

    tracker.record_skipped(3)
    assert tracker.skipped_count == 3

    with pytest.raises(ValueError):
        tracker.set_active_group(-1)
    with pytest.raises(ValueError):
        tracker.set_active_group(1 << 128)
    with pytest.raises(ValueError):
        tracker.set_active_group(True)  # type: ignore[arg-type]
