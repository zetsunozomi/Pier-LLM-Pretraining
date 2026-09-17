"""Executable FP32 specification, not a GPU runtime or performance benchmark.

The protocol partitions learners into contiguous power-of-two cohorts. Within
one cohort, raw local weights are routed to the reference-coordinate owner.
Centering precedes all additions. Cohort partial sums enter the upper levels
of the SAME global balanced addition tree. Outer state update stays at one
coordinate owner, and results are distributed without redoing arithmetic.

Arrays in this specification model all ranks in one Python process. They are
not a measured GPU-memory implementation. Byte counts describe the unpadded,
balanced logical schedule; they are not timings or physical NIC counters.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def power2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def add_tree(values: np.ndarray) -> np.ndarray:
    """Adjacent ordered pairs, repeatedly; no reassociation or numpy.sum."""
    values = np.asarray(values, dtype=np.float32)
    if not power2(values.shape[0]):
        raise ValueError('tree leaf count must be a positive power of two')
    level = values.copy()
    while level.shape[0] > 1:
        level = np.add(level[0::2], level[1::2], dtype=np.float32)
    return level[0].copy()


def update(reference, momentum, average, mu, eta):
    mu, eta = np.float32(mu), np.float32(eta)
    new_m = np.add(np.multiply(mu, momentum, dtype=np.float32), average, dtype=np.float32)
    direction = np.add(average, np.multiply(mu, new_m, dtype=np.float32), dtype=np.float32)
    new_r = np.subtract(reference, np.multiply(eta, direction, dtype=np.float32), dtype=np.float32)
    return new_r, new_m


def validate(reference, momentum, local, cohort):
    if local.dtype != np.float32 or reference.dtype != np.float32 or momentum.dtype != np.float32:
        raise ValueError('spec supports FP32 only')
    if local.ndim != 2 or reference.ndim != 1 or reference.size == 0:
        raise ValueError('nonempty vectors and K x N local masters required')
    k, n = local.shape
    if reference.shape != (n,) or momentum.shape != (n,):
        raise ValueError('coordinate shapes differ')
    if not power2(k) or not power2(cohort) or k % cohort:
        raise ValueError('learner count and cohort size must be nested powers of two')
    if not all(np.isfinite(x).all() for x in (reference, momentum, local)):
        raise ValueError('finite inputs required in this specification')


def resident_reference(reference, momentum, local, mu=.9, eta=.7):
    validate(reference, momentum, local, 1)
    delta = np.subtract(reference[None, :], local, dtype=np.float32)
    average = np.divide(add_tree(delta), np.float32(len(local)), dtype=np.float32)
    new_r, new_m = update(reference, momentum, average, mu, eta)
    return {'average': average, 'reference': new_r, 'momentum': new_m}


def routed_reference(reference, momentum, local, cohort, mu=.9, eta=.7, tile_elements=7):
    validate(reference, momentum, local, cohort)
    if tile_elements < 1:
        raise ValueError('tile_elements must be positive')
    k, n = local.shape
    groups = k // cohort
    # Padding makes the momentum ownership map integral. Padding never enters
    # another coordinate's reduction; final outputs omit the padded suffix.
    width = (n + k - 1) // k
    n_pad = width * k
    r_pad = np.pad(reference, (0, n_pad - n))
    m_pad = np.pad(momentum, (0, n_pad - n))
    w_pad = np.pad(local, ((0, 0), (0, n_pad - n)))
    average = np.empty(n_pad, dtype=np.float32)
    out_r = np.empty(n_pad, dtype=np.float32)
    out_m = np.empty(n_pad, dtype=np.float32)
    # rank (a, b) owns R segment b (one copy in every cohort a), and
    # M segment b*groups + a (one copy globally).
    reference_span = groups * width
    largest_receive_bytes = 0
    for b in range(cohort):
        region_lo, region_hi = b * reference_span, (b + 1) * reference_span
        for lo in range(region_lo, region_hi, tile_elements):
            hi = min(lo + tile_elements, region_hi)
            partials = []
            for a in range(groups):
                ranks = slice(a * cohort, (a + 1) * cohort)
                # Raw-W all-to-all delivery to rank (a, b), leaf-rank order.
                received = w_pad[ranks, lo:hi].copy()
                largest_receive_bytes = max(largest_receive_bytes, received.nbytes)
                # Every subtraction uses the identical R coordinate, locally.
                centered = np.subtract(r_pad[None, lo:hi], received, dtype=np.float32)
                partials.append(add_tree(centered))
            # Corresponding owners run upper tree levels, then divide once.
            reduced = add_tree(np.stack(partials))
            d = np.divide(reduced, np.float32(k), dtype=np.float32)
            rr, mm = update(r_pad[lo:hi], m_pad[lo:hi], d, mu, eta)
            average[lo:hi], out_r[lo:hi], out_m[lo:hi] = d, rr, mm
    if n_pad > n:
        assert (out_r[n:] == 0).all() and (out_m[n:] == 0).all()
    return {'average': average[:n], 'reference': out_r[:n], 'momentum': out_m[:n],
            'padded_elements': n_pad, 'largest_modeled_receive_bytes': largest_receive_bytes,
            'logical_reference_bytes_per_rank': reference_span * 4,
            'logical_momentum_bytes_per_rank': width * 4}


def byte_model(k: int, s: int, p: int):
    if not power2(k) or not power2(s) or k % s or p % k:
        raise ValueError('balanced byte model requires nested powers and P divisible by K')
    g = k // s
    local_a2a = p * (s - 1) // s
    upper_rs = (p // s) * (g - 1) // g
    upper_ag = upper_rs
    local_ag = local_a2a
    stages = [local_a2a, upper_rs, upper_ag, local_ag]
    total = sum(stages)
    assert total == 2 * p * (k - 1) // k
    return {'K': k, 's': s, 'g': g, 'P': p,
            'reference_per_rank': p // s, 'momentum_per_rank': p // k,
            'raw_a2a_sent_per_rank': local_a2a, 'upper_rs_sent_per_rank': upper_rs,
            'upper_ag_sent_per_rank': upper_ag, 'local_ag_sent_per_rank': local_ag,
            'total_sent_per_rank': total}


def u32(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def counterexample():
    r = np.array([1.0], dtype=np.float32)
    m = np.array([0.0], dtype=np.float32)
    w = np.array([[1.0], [1.0 + 2.0**-23]], dtype=np.float32)
    ref = resident_reference(r, m, w)
    average_w = np.divide(add_tree(w), np.float32(2), dtype=np.float32)
    shifted = np.subtract(r, average_w, dtype=np.float32)
    shifted_r, shifted_m = update(r, m, shifted, .9, .7)
    assert ref['average'][0] == np.float32(-2.0**-24)
    assert shifted[0] == 0
    assert u32(ref['reference'])[0] != u32(shifted_r)[0]
    return {'kind': 'constructed_FP32_witness_not_training_measurement',
            'R': r.tolist(), 'W': w.tolist(), 'centered_average': ref['average'].tolist(),
            'recentered_average': shifted.tolist(), 'centered_master': ref['reference'].tolist(),
            'recentered_master': shifted_r.tolist(),
            'centered_master_bits': [hex(int(v)) for v in u32(ref['reference'])],
            'recentered_master_bits': [hex(int(v)) for v in u32(shifted_r)]}


def run_checks():
    rng = np.random.default_rng(20260915)
    cases, compared = 0, 0
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        for n in (1, 3, 17, 67):
            initial_r = rng.normal(0, .2, n).astype(np.float32)
            initial_m = rng.normal(0, .01, n).astype(np.float32)
            r, m = initial_r, initial_m
            for round_id in range(3):
                scale = (1e-1, 1e-4, 1e-7)[round_id]
                w = np.add(r[None, :], rng.normal(0, scale, (k,n)).astype(np.float32), dtype=np.float32)
                expected = resident_reference(r, m, w)
                for s in [2**j for j in range(k.bit_length())]:
                    got = routed_reference(r,m,w,s,tile_elements=(1,7,19)[round_id])
                    for key in ('average','reference','momentum'):
                        if not np.array_equal(u32(expected[key]),u32(got[key])):
                            raise AssertionError((k,n,s,round_id,key))
                        compared += n
                    assert got['largest_modeled_receive_bytes'] <= s*(1,7,19)[round_id]*4
                    cases += 1
                r, m = expected['reference'], expected['momentum']
    # Cancellation and signed-zero cases explicitly exercise ordered additions.
    w = np.array([[2**24,-0.0],[1,0.0],[-2**24,-0.0],[1,0.0]],np.float32)
    r = np.zeros(2,np.float32); m = np.zeros(2,np.float32)
    for s in (1,2,4):
        e,g = resident_reference(r,m,w),routed_reference(r,m,w,s,tile_elements=1)
        for key in ('average','reference','momentum'):
            assert np.array_equal(u32(e[key]),u32(g[key]))
    invalid_rejected = 0
    for k,s in ((3,1),(4,3),(4,8)):
        try:
            routed_reference(np.zeros(2,np.float32),np.zeros(2,np.float32),np.zeros((k,2),np.float32),s)
        except ValueError:
            invalid_rejected += 1
    assert invalid_rejected == 3
    counts=[byte_model(8,s,8*1024*1024) for s in (1,2,4,8)]
    return {'status':'passed', 'backend':'single-process NumPy CPU specification',
            'parameterized_cases':cases,'compared_FP32_coordinates':compared,
            'extra_adversarial_cases':3,'unsupported_shapes_rejected':invalid_rejected,
            'scope':'ordered outer arithmetic, nested cohort equivalence, padding, logical byte equations',
            'not_tested':['GPU runtime','distributed transport','kernel determinism','concurrent buffer lifetime','training throughput','model quality'],
            'counterexample':counterexample(),'analytical_byte_examples':counts,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('verification.json'))
    args=parser.parse_args()
    result=run_checks()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','backend','parameterized_cases','compared_FP32_coordinates','extra_adversarial_cases','unsupported_shapes_rejected')},indent=2))
