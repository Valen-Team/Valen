import random
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from types import SimpleNamespace

import pytest

from valen.data.prefetch import PackPrefetcher


class Compiler:
    def compile(self, record, rng, labeled_only):
        candidates = list(range(5))
        rng.shuffle(candidates)
        return SimpleNamespace(compute_tokens=record["tokens"], questions=[candidates], media=[])


class Backend:
    def validate_state(self, config, compiled):
        pass


def collect(records, enabled, cursor=0, rng_state=None):
    order = list(range(len(records)))
    rng = random.Random(17)
    if rng_state is not None:
        rng.setstate(rng_state)
    output = []
    with PackPrefetcher(records, Compiler(), Backend(),
                        {"tokens_per_step": 7, "data_prefetch": enabled}) as source:
        while cursor < len(order):
            pack = source.take(order, cursor, rng.getstate())
            cursor = pack.cursor
            rng.setstate(pack.rng_state)
            output.append((pack.cursor, pack.tokens, [s.questions for s in pack.states], pack.rng_state))
    return output


def test_prefetch_and_resume_keep_exact_question_order_rng_and_token_packs():
    records = [{"group_id": str(i), "tokens": tokens} for i, tokens in enumerate([3, 4, 2, 5, 4, 2, 1])]
    serial = collect(records, False)
    prefetched = collect(records, True)
    assert prefetched == serial
    # The next pack was already queued when this checkpoint cursor was committed.
    cursor, _, _, rng_state = prefetched[0]
    assert collect(records, True, cursor, rng_state) == serial[1:]
    assert serial[-1][0] == len(records)


def test_prefetch_detects_a_changed_cursor_without_consuming_the_wrong_pack():
    records = [{"group_id": str(i), "tokens": 4} for i in range(3)]
    order = list(range(3))
    with PackPrefetcher(records, Compiler(), Backend(),
                        {"tokens_per_step": 7, "data_prefetch": True}) as source:
        result = source.take(order, 0, random.Random(1).getstate())
        with pytest.raises(ValueError, match="committed cursor/RNG"):
            source.take(order, result.cursor + 1, result.rng_state)


def test_slow_prefetch_has_longer_bounded_wait_and_reports_active_record():
    class SlowFuture:
        def result(self, timeout):
            assert timeout == 540
            raise FutureTimeoutError()

        def done(self):
            return False

    order, rng_state = [0], random.Random(1).getstate()
    with PackPrefetcher([], Compiler(), Backend(), {"tokens_per_step": 7}) as source:
        source.pending = ((id(order), 0, rng_state), SlowFuture())
        source.active_record = (23, "slow-video")
        with pytest.raises(TimeoutError, match=r"Data prefetch.*cursor=0.*slow-video"):
            source.take(order, 0, rng_state)


def test_prefetch_preserves_timeout_raised_by_compiler():
    order, rng_state = [0], random.Random(1).getstate()
    future = Future()
    future.set_exception(TimeoutError("media decoder timed out"))
    with PackPrefetcher([], Compiler(), Backend(), {"tokens_per_step": 7}) as source:
        source.pending = ((id(order), 0, rng_state), future)
        with pytest.raises(TimeoutError, match="^media decoder timed out$"):
            source.take(order, 0, rng_state)


def test_a_rejected_record_is_skipped_not_raised():
    """One bad record must not raise: a raising rank strands its peers in a collective."""
    import valen.data.prefetch as prefetch_module

    class Rejecting:
        def __init__(self):
            self.calls = 0

        def compile(self, record, rng=None, labeled_only=False):
            self.calls += 1
            if record["group_id"] == "bad":
                raise ValueError("Reserved tokenizer control token in question/criteria")
            return SimpleNamespace(questions=[object()], compute_tokens=10, media=[], inputs=None)

    records = [{"group_id": name} for name in ("ok1", "bad", "ok2")]
    compiler = Rejecting()
    backend = SimpleNamespace(validate_state=lambda config, state: None)
    config = {"tokens_per_step": 1000, "data_prefetch": False}
    with prefetch_module.PackPrefetcher(records, compiler, backend, config) as source:
        prepared = source.take([0, 1, 2], 0, random.Random(0).getstate())
        assert len(prepared.states) == 2, "the good records must still be packed"
        assert prepared.cursor == 3
        assert len(source.skipped_records) == 1
        assert "Reserved tokenizer control token" in source.skipped_records[0]["error"]
