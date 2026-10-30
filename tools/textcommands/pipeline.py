#!/usr/bin/env python3
"""From a command list to a program that does what it is told: the whole pipeline.

1. generate.py writes training sentences at scale, and a test set of phrasings training never sees.
2. logievo_run --train trains a Logievo project on them (the Text Commands template the first
   time, its last layer sized to the commands) and exports it as an executable.
3. The executable is checked on new sentences, on phrasings it never saw, then on a few sentences.

    python tools/textcommands/pipeline.py --count 200000 --epochs 6

The project (open it in Logievo to look at it or train more), the examples and the program go in
--out-dir. Run again to train further from what the project learned; --fresh starts over.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import generate

DEMO = [
    "abre a calculadora pra mim",
    "por favor abra o bloco de notas",
    "Ei, mostra a pasta de downloads.",
    "abri o terminal",  # Whisper's "abri" for "abre"
    "não abra a calculadora",
    "feche o paint",
    "abra o spotify",
    "abra a porta",
    "desliga o computador",
    "destrói o terminal",
    "calculadora",
    "open the task manager",
]


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    exe = ".exe" if sys.platform == "win32" else ""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--commands", type=Path, default=here / "commands.txt")
    p.add_argument("--out-dir", type=Path, default=root / "build" / "textcommands")
    p.add_argument("--name", default="comandos", help="the project's and the program's name")
    p.add_argument("--runner", type=Path, default=root / "build" / f"logievo_run{exe}")
    p.add_argument("--count", type=int, default=200_000, help="training sentences")
    p.add_argument("--test-count", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=128, help="examples per step: bigger is faster, the weights are many")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--noise", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fresh", action="store_true", help="forget what the project learned before")
    a = p.parse_args(argv)

    if not a.runner.exists():
        sys.exit(f"{a.runner} is missing: build Logievo first (run.bat --build-only)")
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    project, program = out / f"{a.name}.logievo", out / f"{a.name}{exe}"
    train, test, known = out / "train.tsv", out / "test.tsv", out / "known.tsv"
    if a.fresh and project.exists():
        project.unlink()

    print("== generating sentences", flush=True)
    generate.main(["--commands", str(a.commands), "--out", str(train), "--count", str(a.count), "--seed", str(a.seed), "--noise", str(a.noise)])
    generate.main(["--commands", str(a.commands), "--out", str(test), "--count", str(a.test_count), "--seed", str(a.seed + 1000),
                   "--noise", str(a.noise), "--split", "test", "--unique"])
    generate.main(["--commands", str(a.commands), "--out", str(known), "--count", str(a.test_count), "--seed", str(a.seed + 2000),
                   "--noise", str(a.noise), "--unique"])

    print("== training", flush=True)
    subprocess.run([str(a.runner), "--train", str(project), "--data", str(train), "--epochs", str(a.epochs), "--batch", str(a.batch),
                    "--lr", str(a.lr), "--export", str(program)], check=True)

    print("== new sentences, phrasings it was trained on", flush=True)
    subprocess.run([str(program), "--eval", str(known)], check=True)
    print("== phrasings it never saw (the hard test: wording that changes the meaning is to be added to generate.py)", flush=True)
    subprocess.run([str(program), "--eval", str(test)], check=True)

    print("== a few sentences (--dry-run: nothing is opened)", flush=True)
    for s in DEMO:
        r = subprocess.run([str(program), "--dry-run", s], capture_output=True, text=True, encoding="utf-8")
        print(f"  {s!r:45} -> {r.stdout.strip()}")
    print(f"\n{program} is ready:  {program.name} \"abra a calculadora\"   (or pipe Whisper's text into it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
