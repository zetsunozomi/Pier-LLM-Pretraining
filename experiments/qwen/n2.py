#!/usr/bin/env python3
"""Launch selected outer-state arms on one allocation and publish each result."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.n2_config import configuration, cases, launch_command, training_args


def source_identity():
    command = ['git', 'ls-files', '-z', '--', '*.py', '*.sh', '*.sbatch', 'experiments/qwen/pins.json']
    paths = subprocess.check_output(command, cwd=ROOT).decode().split('\0')
    # Include newly implemented N2 files even before the user's first commit.
    paths += [str(p.relative_to(ROOT)) for pattern in ('n2*', 'n3*')
              for p in (ROOT / 'experiments/qwen').glob(pattern) if p.is_file()]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / 'megatron/core/outer_sync').glob('*.py')]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in sorted(set(paths)) if name and (ROOT / name).is_file()}
    return hashes


def run(output):
    cfg = configuration()
    stage = cfg.get('stage', 'N2')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    if path.exists():
        raise FileExistsError('N2 requires a fresh output directory')
    pin = json.loads((ROOT / 'experiments/qwen/pins.json').read_text())['models']['3B']
    for name, record in pin['files'].items():
        target = Path(cfg['snapshot']) / name
        if not target.is_file() or target.stat().st_size != record['bytes']:
            raise ValueError(f'missing or incomplete snapshot file: {target}')
    if cfg['data_prefix']:
        for suffix in ('.bin', '.idx', '.manifest.json'):
            if not Path(cfg['data_prefix'] + suffix).is_file():
                raise ValueError(f'missing indexed input: {cfg["data_prefix"] + suffix}')
    manifest = {'stage': stage, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'config': cfg, 'cases': cases(cfg), 'sources': source_identity(), 'output_directory': str(output),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'E0c_required': False}
    # Freeze the actual commands so later recipe edits cannot invalidate old runs.
    manifest['training_argv'] = {case['id']: training_args(cfg, case, output / case['id'])
                                 for case in manifest['cases']}
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    from experiments.qwen.n2_summary import save_summary
    print(json.dumps(cfg, indent=2), flush=True)
    print(f'{stage} artifacts: {output}\nEach arm starts from the same pretrained Qwen snapshot.', flush=True)
    if stage == 'N3':
        from experiments.qwen.n2_summary import copy_historical_reference
        reference = os.environ.get('PIER_N3_REFERENCE')
        if reference:
            copy_historical_reference(Path(reference), output)
    print('Pilot: 50 warmup steps + 50 measured steps per arm.' if cfg['profile'] == 'pilot'
          else 'Main window: 100 warmup steps + 150 measured steps per arm.', flush=True)
    interrupted = False
    try:
        for case in manifest['cases']:
            if source_identity() != manifest['sources']:
                raise RuntimeError('source changed during N2; start a fresh run')
            directory = output / case['id']
            directory.mkdir()
            command = launch_command(cfg, case, output, slurm=bool(os.environ.get('SLURM_JOB_ID')))
            print(f"[{stage}] START {case['id']}", flush=True)
            started = time.monotonic()
            with (output / f"{case['id']}.log").open('w') as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    try:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end='', flush=True)
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise
            (directory / 'exit.json').write_text(json.dumps({
                'exit_code': code, 'wall_seconds_including_startup': time.monotonic() - started,
                'command': command}, indent=2) + '\n')
            print(f"[{stage}] DONE {case['id']}: exit={code}", flush=True)
            save_summary(output)
    except BaseException:
        interrupted = True
        raise
    finally:
        result = save_summary(output)
        print(f'{stage} results: {output}/results.txt\n{stage} JSON: {output}/summary.json', flush=True)
    return 0 if not interrupted and result['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.output_dir.resolve()))
