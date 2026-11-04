"""Prepare the next token pack while the GPU trains. / CPU 预处理与 GPU 计算重叠。"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import AbstractContextManager
from dataclasses import dataclass
import random
import time


@dataclass
class PreparedPack:
    states: list
    tokens: int
    cursor: int
    rng_state: tuple
    media: list
    prepare_seconds: float


class PackPrefetcher(AbstractContextManager):
    def __init__(self, records, compiler, backend, config, on_compile=None):
        self.records, self.compiler, self.backend, self.config = records, compiler, backend, config
        self.budget = config["tokens_per_step"]
        self.on_compile = on_compile
        self.executor = ThreadPoolExecutor(max_workers=1) if config.get("data_prefetch", False) else None
        self.pending = None
        self.cached_overflow = None
        self.active_record = None
        self.skipped = []

    def _prepare(self, order, cursor, rng_state):
        started = time.monotonic()
        rng = random.Random()
        rng.setstate(rng_state)
        pack, tokens, media = [], 0, []
        while cursor < len(order):
            index = order[cursor]
            self.active_record = (index, self.records[index]["group_id"])
            before = rng.getstate()
            if self.on_compile:
                self.on_compile(index, self.records[index]["group_id"])
            cached = self.cached_overflow
            if cached and cached[:2] == (index, before):
                compiled = cached[2]
                rng.setstate(cached[3])
                self.cached_overflow = None
            else:
                try:
                    compiled = self.compiler.compile(self.records[index], rng, labeled_only=True)
                except (ValueError, KeyError, TypeError) as exc:
                    # A raise here would leave this rank's peers waiting in the next
                    # collective until the NCCL timeout kills the job, so skip the record
                    # and record it instead. Validate a mix up front with
                    # valen-next/scripts/find_bad_records.py.
                    # 此处抛异常会让本 rank 退出，其余 rank 在集合通信里等到 NCCL 超时，
                    # 整个任务因一条数据而死；改为跳过并记录。
                    rng.setstate(before)
                    self.skipped.append({"index": index, "error": str(exc)})
                    cursor += 1
                    continue
            if compiled.compute_tokens > self.budget:
                raise ValueError(f"State {index} exceeds tokens_per_step: {compiled.compute_tokens} > {self.budget}")
            if pack and tokens + compiled.compute_tokens > self.budget:
                self.cached_overflow = (index, before, compiled, rng.getstate())
                rng.setstate(before)
                break
            cursor += 1
            if not compiled.questions:
                continue
            self.backend.validate_state(self.config, compiled)
            pack.append(compiled)
            tokens += compiled.compute_tokens
            media.append({"state_index": index, "group_id": self.records[index]["group_id"],
                          "media": compiled.media})
        return PreparedPack(pack, tokens, cursor, rng.getstate(), media, time.monotonic() - started)

    @property
    def skipped_records(self):
        """Records the compiler rejected, skipped to keep the collectives aligned."""
        return list(self.skipped)

    def take(self, order, cursor, rng_state, prefetch_next=True):
        # Only returned packs become committed checkpoint cursor/RNG state.
        # 预取中的游标和 RNG 不写入 checkpoint，恢复时仍从已完成的数据开始。
        key = (id(order), cursor, rng_state)
        if self.pending is not None:
            previous_key, future = self.pending
            if previous_key != key:
                raise ValueError("Prefetched pack does not match the committed cursor/RNG")
            # Large video packs can exceed five minutes during storage contention.
            # 视频大包在存储繁忙时可能超过五分钟，仍保留低于 NCCL 十分钟的等待上限。
            try:
                result = future.result(timeout=540)
            except FutureTimeoutError as error:
                # A compiler-raised TimeoutError must retain its original context.
                # 保留编译器自身异常，仅为尚未完成的预取增加定位信息。
                if future.done():
                    result = future.result()
                else:
                    raise TimeoutError(
                        f"Data prefetch timed out after 540 seconds: cursor={cursor}, "
                        f"active_record={self.active_record}"
                    ) from error
            self.pending = None
        else:
            result = self._prepare(order, cursor, rng_state)
        if self.executor and prefetch_next and result.cursor < len(order):
            key = (id(order), result.cursor, result.rng_state)
            self.pending = (key, self.executor.submit(self._prepare, order, result.cursor, result.rng_state))
        return result

    def __exit__(self, *args):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
