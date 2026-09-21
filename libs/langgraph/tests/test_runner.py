"""Unit coverage for the runner-level stop predicate used on task completion.

Regression for langchain-ai/langgraph#8859: `FuturesDict.on_done` previously
rescanned the whole accumulated `done` set through `_should_stop_others` on every
completion, making a superstep O(tasks^2). Because a completed future's failure
state is immutable and each future is checked when it completes, only the
newly-completed future can newly trigger the stop, so the per-completion check is
now O(1) via `_is_stopping_failure`. These tests pin the behavioral contract of
that predicate and its equivalence with the batch scan.
"""

import concurrent.futures
import threading
import weakref
from functools import partial

from langgraph.errors import GraphInterrupt
from langgraph.pregel._runner import (
    SKIP_RERAISE_SET,
    FuturesDict,
    _is_stopping_failure,
    _should_stop_others,
)


def _settled(result=None, exc=None):
    fut: concurrent.futures.Future = concurrent.futures.Future()
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(result)
    return fut


def _cancelled():
    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.cancel()
    fut.set_running_or_notify_cancel()
    return fut


def test_success_is_not_a_stopping_failure():
    assert _is_stopping_failure(_settled(result="ok")) is False


def test_real_exception_is_a_stopping_failure():
    assert _is_stopping_failure(_settled(exc=ValueError("boom"))) is True


def test_cancelled_is_not_a_stopping_failure():
    assert _is_stopping_failure(_cancelled()) is False


def test_graph_interrupt_is_not_a_stopping_failure():
    # GraphInterrupt is a GraphBubbleUp; interrupts are control flow, not failures.
    assert _is_stopping_failure(_settled(exc=GraphInterrupt())) is False


def test_handled_exception_id_is_not_a_stopping_failure():
    exc = ValueError("handled")
    fut = _settled(exc=exc)
    assert _is_stopping_failure(fut, handled_exception_ids={id(exc)}) is False
    # A different handled id does not suppress this one.
    assert _is_stopping_failure(fut, handled_exception_ids={id(object())}) is True


def test_skip_reraise_set_future_is_not_a_stopping_failure():
    fut = _settled(exc=ValueError("skip"))
    SKIP_RERAISE_SET.add(fut)
    try:
        assert _is_stopping_failure(fut) is False
    finally:
        SKIP_RERAISE_SET.discard(fut)


def test_batch_scan_matches_incremental_check_across_the_done_set():
    # The incremental per-completion decision must agree with a full scan of the
    # accumulated set: the set stops iff any member individually stops.
    ok = [_settled(result=i) for i in range(5)]
    interrupt = _settled(exc=GraphInterrupt())
    cancelled = _cancelled()

    all_benign = {*ok, interrupt, cancelled}
    assert _should_stop_others(all_benign) is False
    assert not any(_is_stopping_failure(f) for f in all_benign)

    failing = _settled(exc=ValueError("boom"))
    with_failure = {*all_benign, failing}
    assert _should_stop_others(with_failure) is True
    # Only the failing future flips the answer; the benign ones stay False, which
    # is exactly why checking just the newly-completed future is sufficient.
    assert _is_stopping_failure(failing) is True
    assert all(not _is_stopping_failure(f) for f in all_benign)


def _make_futures_dict():
    # callback() returns None (weakref to a dropped object) so on_done's
    # `if cb := self.callback()` branch is a no-op; we drive it via completions.
    dead_ref: weakref.ref = weakref.ref(threading.Lock())
    return FuturesDict(
        event=threading.Event(),
        callback=dead_ref,  # type: ignore[arg-type]
        should_stop=partial(_is_stopping_failure, handled_exception_ids=None),
        future_type=concurrent.futures.Future,
    )


# A non-None value makes __setitem__ track the future (increment counter, clear
# the event, register on_done). The stored value is only ever handed to the
# callback, which is a no-op in these tests, so a sentinel stands in for a real
# PregelExecutableTask.
_TASK = object()


def _register(fd, fut):
    # Registering a still-pending future clears the event and increments the
    # counter; completing it later fires on_done synchronously, exactly as the
    # runtime does when a task finishes.
    fd[fut] = _TASK  # type: ignore[assignment]


def test_stop_latches_across_a_later_registration_that_clears_the_event():
    # Regression for the level-triggered behavior the full-set rescan used to
    # provide: a fatal completion sets the event; a task then dynamically
    # registers a new future, whose __setitem__ clears the event; a benign
    # completion must still leave the event set so the runner reaches its
    # panic/cancel path instead of blocking until every task finishes.
    fd = _make_futures_dict()

    # A future that is still pending when registered (so on_done does not fire at
    # registration); we complete it explicitly to control ordering.
    fatal: concurrent.futures.Future = concurrent.futures.Future()
    _register(fd, fatal)  # counter == 1, event cleared, callback pending
    assert fd.event.is_set() is False

    fatal.set_exception(ValueError("boom"))  # fires on_done synchronously
    assert fd.stop_requested is True
    assert fd.event.is_set() is True

    # A running task dynamically registers another (still pending) future ->
    # __setitem__ clears the event; the latch must re-assert it.
    benign: concurrent.futures.Future = concurrent.futures.Future()
    _register(fd, benign)
    assert fd.event.is_set() is True  # re-asserted because a stop already fired

    # A benign completion afterwards keeps the stop latched.
    benign.set_result("ok")  # fires on_done synchronously
    assert fd.stop_requested is True
    assert fd.event.is_set() is True


def test_no_stop_when_all_futures_succeed_across_registrations():
    # Symmetric control: without any failure, the event is only set once every
    # tracked future is done (counter == 0), never spuriously latched.
    fd = _make_futures_dict()

    first: concurrent.futures.Future = concurrent.futures.Future()
    _register(fd, first)  # counter == 1
    second: concurrent.futures.Future = concurrent.futures.Future()
    _register(fd, second)  # counter == 2

    first.set_result(1)  # fires on_done -> counter == 1
    assert fd.stop_requested is False
    assert fd.event.is_set() is False  # not all done, no failure

    second.set_result(2)  # fires on_done -> counter == 0
    assert fd.stop_requested is False
    assert fd.event.is_set() is True  # all done
