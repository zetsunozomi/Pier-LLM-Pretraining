"""Deterministic layouts and per-rank admission for the reference-refresh runtime.

Reference allocation pages and momentum homes have stable coordinate identities.
Execution tiles may combine pages, so changing concurrency or tile size never
requires a state redistribution. Bytes here are explicit tensor bytes; callers
must supply their other live allocations and allocator/library headroom.
"""

from dataclasses import asdict, dataclass
import math


class AdmissionError(MemoryError):
    """A rejected plan, before any outer state or model value is modified."""


def power2(value):
    return isinstance(value, int) and value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class Layout:
    learners: int
    cohort: int

    def __post_init__(self):
        if not power2(self.learners) or not power2(self.cohort) or self.learners % self.cohort:
            raise ValueError('nested power-of-two learners/cohort required')

    @property
    def groups(self):
        return self.learners // self.cohort

    def owners(self, shard):
        if not 0 <= shard < self.learners:
            raise ValueError('shard outside layout')
        b = shard // self.groups
        return tuple(a * self.cohort + b for a in range(self.groups))

    def executor(self, shard):
        return self.owners(shard)[shard % self.groups]

    def owns(self, rank, shard):
        return rank in self.owners(shard)


@dataclass(frozen=True)
class Tile:
    shard: int
    offset: int
    count: int


def tiles(learners, width, capacity):
    # Interleave owners to avoid retiring one rank's entire reference first.
    return tuple(Tile(j, offset, min(capacity, width - offset))
                 for offset in range(0, width, capacity) for j in range(learners))


def slot_bytes(cohort, capacity):
    # W/send/direction, received/sum/return, R, M, binary carry stack.
    return 4 * capacity * (4 + (cohort.bit_length() - 1))


@dataclass(frozen=True)
class Plan:
    old_cohort: int
    cohort: int
    capacity: int
    slots: int
    mode: str
    peaks: tuple
    final_bytes: tuple
    workspace_bytes: int
    remote_momentum_bytes: tuple
    estimate_seconds: float
    cost_source: str

    def receipt(self):
        return asdict(self)


def memory_schedule(k, width, old, new, capacity, slots, *, mode='joint', base=None, headroom=0):
    """Exact explicit-storage envelope of the wave schedule used by the executor.

    A wave issues q independent tiles. In deterministic tile order it drains
    old consumers, reuses/frees pages, admits destinations and issues returns.
    Independent composition refreshes the old layout, drains it, drops it, and
    constructs the new layout from committed masters in a separate local pass.
    It is not penalized with an artificial full-old-plus-full-new allocation.
    """
    if mode not in ('joint', 'separate', 'static') or min(width, capacity, slots) < 1:
        raise ValueError('invalid execution mode/size')
    if mode == 'static' and old != new:
        raise ValueError('static execution cannot change layout')
    before, after = Layout(k, old), Layout(k, new)
    base = tuple(base or (0,) * k)
    if len(base) != k or min(base) < 0 or headroom < 0:
        raise ValueError('one nonnegative base allocation per rank required')
    workspace = slots * slot_bytes(old, capacity)
    fixed = [x + headroom + width * 4 + workspace for x in base]
    resident = [width * 4 * before.groups] * k
    peaks = [fixed[i] + resident[i] for i in range(k)]
    if mode == 'joint':
        # The tile sequence repeats the same shard order at every offset.
        # Within one stripe, the prefix ownership delta determines the peak;
        # across full stripes its drift is linear. Evaluate the extremal full
        # stripe and the optional tail exactly, without walking model-sized
        # tile lists for every candidate and planning horizon.
        delta, prefix_peak = [0] * k, [0] * k
        for shard in range(k):
            for i in before.owners(shard):
                delta[i] -= 1
            for i in after.owners(shard):
                delta[i] += 1
            prefix_peak = [max(prefix_peak[i], delta[i]) for i in range(k)]
        full, tail = divmod(width, capacity)
        for i in range(k):
            changes = [0]
            if full:
                stripe = full - 1 if delta[i] > 0 else 0
                changes.append(capacity * (stripe * delta[i] + prefix_peak[i]))
            if tail:
                changes.append(full * capacity * delta[i] + tail * prefix_peak[i])
            peaks[i] += 4 * max(changes)
    final = tuple(base[i] + headroom + width * 4 * (1 + after.groups) for i in range(k))
    # Separate construction runs after scratch and old reference are released.
    peaks = tuple(max(peaks[i], final[i]) for i in range(k))
    remote = [0] * k
    for shard in range(k):
        home, executor = shard, before.executor(shard)
        if home != executor:
            remote[home] += width * 4
            remote[executor] += width * 4
    return peaks, final, workspace, tuple(remote)


class Planner:
    """Enumerate feasible (layout, tile, slots), optionally from measured costs.

    The fallback is an explicitly labeled bandwidth/launch model, not a claim of
    measured latency. Calibration entries are indexed by old/target layout,
    capacity, slots and mode. Missing entries do not borrow a different mode's
    timings. All selection is deterministic given the same per-rank ledger.
    """

    def __init__(self, learners, width, page_elements, *, cohorts=None,
                 capacities=None, slots=(1, 2, 4), costs=(), bandwidth=12e9, launch_seconds=8e-6):
        Layout(learners, 1)
        if min(width, page_elements) < 1 or bandwidth <= 0 or launch_seconds < 0:
            raise ValueError('positive dimensions and valid cost-model parameters required')
        self.k, self.width, self.page = learners, width, page_elements
        self.cohorts = tuple(sorted(set(cohorts or (1, min(2, learners), learners))))
        for cohort in self.cohorts:
            Layout(learners, cohort)
        self.capacities = tuple(sorted(set(capacities or (page_elements, 2 * page_elements, 4 * page_elements))))
        if any(c < page_elements or c % page_elements for c in self.capacities):
            raise ValueError('tile capacities must be positive multiples of allocation page size')
        self.slots = tuple(sorted(set(slots)))
        if not self.slots or min(self.slots) < 1:
            raise ValueError('positive slot counts required')
        self.costs = {}
        for row in costs:
            key = tuple(row[k] for k in ('old_cohort', 'cohort', 'capacity', 'slots', 'mode'))
            value = float(row['seconds'])
            if not math.isfinite(value) or value <= 0 or key in self.costs:
                raise ValueError('unique positive measured configuration costs required')
            self.costs[key] = value
        self.bandwidth, self.launch_seconds = bandwidth, launch_seconds

    def candidates(self, old, budgets, *, base=None, headroom=0, mode='joint',
                   target=None, fixed_slots=None, workspace_limit=None, final_budgets=None):
        if len(budgets) != self.k or any(not math.isfinite(x) or x <= 0 for x in budgets):
            raise ValueError('positive per-rank budgets required')
        final_budgets = budgets if final_budgets is None else final_budgets
        if len(final_budgets) != self.k or any(not math.isfinite(x) or x <= 0 for x in final_budgets):
            raise ValueError('positive next-phase per-rank budgets required')
        choices = (target,) if target is not None else ((old,) if mode == 'static' else self.cohorts)
        result = []
        for cohort in choices:
            for capacity in self.capacities:
                for q in ((fixed_slots,) if fixed_slots is not None else self.slots):
                    if q < 1:
                        raise ValueError('positive slot count required')
                    peaks, final, workspace, remote = memory_schedule(
                        self.k, self.width, old, cohort, capacity, q,
                        mode=mode, base=base, headroom=headroom)
                    if workspace_limit is not None and workspace > workspace_limit:
                        continue
                    if any(peak > limit for peak, limit in zip(peaks, budgets)):
                        continue
                    if any(size > limit for size, limit in zip(final, final_budgets)):
                        continue
                    key = old, cohort, capacity, q, mode
                    if key in self.costs:
                        estimate, source = self.costs[key], 'measured configuration table'
                    else:
                        count = self.k * math.ceil(self.width / capacity)
                        payload = 8 * self.width * (self.k - 1) + max(remote)
                        # Local reference construction makes the extra master
                        # read in separate composition explicit, without adding
                        # nonexistent reference network traffic.
                        local = 4 * self.width * (self.k // cohort) if mode == 'separate' and cohort != old else 0
                        estimate = ((payload + local) / self.bandwidth
                                    + count * (2 * old + 2 * self.k.bit_length()) * self.launch_seconds / q)
                        source = 'analytic bandwidth/launch estimate (uncalibrated)'
                    result.append(Plan(old, cohort, capacity, q, mode, peaks, final, workspace,
                                       remote, estimate, source))
        return sorted(result, key=lambda p: (p.estimate_seconds, max(p.peaks), p.cohort, p.capacity, p.slots))

    def choose(self, old, budgets, *, remaining_rounds=1, **options):
        if remaining_rounds < 1:
            raise ValueError('positive remaining round horizon required')
        feasible = self.candidates(old, budgets, **options)
        if not feasible:
            raise AdmissionError('no progress-safe layout/tile/slot plan fits all rank budgets')
        # Evaluate each transition together with the best feasible steady state.
        # Forced target experiments deliberately exercise every specified edge.
        if options.get('target') is not None or remaining_rounds == 1:
            return feasible[0]
        scored = []
        continuations = {}
        for plan in feasible:
            if plan.cohort not in continuations:
                steady_options = {**options, 'target': plan.cohort}
                continuations[plan.cohort] = self.candidates(plan.cohort, budgets, **steady_options)
            steady = continuations[plan.cohort]
            if steady:
                scored.append((plan.estimate_seconds + (remaining_rounds - 1) * steady[0].estimate_seconds,
                               max(plan.peaks), plan))
        if not scored:
            raise AdmissionError('no candidate has a feasible steady-state continuation')
        return min(scored, key=lambda row: (row[0], row[1], row[2].cohort, row[2].capacity, row[2].slots))[2]
