#!/usr/bin/env python3
"""Launch G/P/R/W on the same allocation and publish results after each arm."""

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
from experiments.qwen.n2_config import configuration, cases, launch_command


def source_identity():
    command = ['git', 'ls-files', '-z', '--', '*.py', '*.sh', '*.sbatch', 'experiments/qwen/pins.json']
    paths = subprocess.check_output(command, cwd=ROOT).decode().split('\0')
    # Include newly implemented N2 files even before the user's first commit.
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / 'experiments/qwen').glob('n2*') if p.is_file()]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in sorted(set(paths)) if name and (ROOT / name).is_file()}
    return hashes


def run(output):
    cfg = configuration()
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
    manifest = {'stage': 'N2', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'config': cfg, 'cases': cases(cfg), 'sources': source_identity(), 'output_directory': str(output),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'E0c_required': False}
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    from experiments.qwen.n2_summary import save_summary
    print(json.dumps(cfg, indent=2), flush=True)
    print(f'N2 artifacts: {output}\nEach arm starts from the same pretrained Qwen snapshot.', flush=True)
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
            print(f"[N2] START {case['id']}", flush=True)
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
            print(f"[N2] DONE {case['id']}: exit={code}", flush=True)
            save_summary(output)
    except BaseException:
        interrupted = True
        raise
    finally:
        result = save_summary(output)
        print(f'N2 results: {output}/results.txt\nN2 JSON: {output}/summary.json', flush=True)
    return 0 if not interrupted and result['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.output_dir.resolve()))
