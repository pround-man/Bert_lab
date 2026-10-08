"""Local commands. check/verify/report are offline; only build/smoke call models."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock, Timeout

from .config import load_catalog, load_settings
from .migrations import upgrade_dataset
from .models import DatasetError
from .pipeline import run_build, state_directory, verify_dataset
from .planning import plan_slots
from .provider import LLMProvider
from .quality import TextInspector
from .reports import build_report
from .runtime import load_runtime
from .storage import Store, atomic_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build synthetic en/zh-Hans/th watch-intent datasets.")
    result.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: cwd)")
    result.add_argument("--config", type=Path, help="TOML config path (relative to repository root)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Offline config/credential-presence/model/plan validation")
    commands.add_parser("upgrade", help="Offline backup and migration of the known legacy checkpoint")
    commands.add_parser("smoke", help="Paid API smoke: at most 9 requested candidates, no retries")
    build = commands.add_parser("build", help="Paid generation + blind review until all quotas are filled")
    build.add_argument("--stage", choices=("pilot", "full"), required=True)
    build.add_argument("--resume", action="store_true", help="Reuse existing accepted checkpoints")
    build.add_argument("--bounded", action="store_true",
                       help="Stop at the original per-slot attempt limit instead of continuous generation")
    build.add_argument("--generator-workers", type=int, help="Override generator concurrency from concurrent.env")
    build.add_argument("--judge-workers", type=int, help="Override reviewer concurrency from concurrent.env")
    for name in ("verify", "report"):
        command = commands.add_parser(name, help="Offline dataset " + name)
        command.add_argument("--stage", choices=("pilot", "full", "smoke"), required=True)
    return result


def _print(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def execute(args) -> int:
    settings = load_settings(args.root, args.config)
    catalog = load_catalog(settings.root)
    if args.command == "report":
        slots = plan_slots(settings, catalog, args.stage)
        directory = state_directory(settings, args.stage)
        try:
            with FileLock(str(directory / ".run.lock"), timeout=0), Store(directory, None, resume=True) as store:
                if store.get_meta("identity")["settings"] != settings.fingerprint_payload():
                    raise DatasetError("Report config differs from dataset; use its original config.")
                report = build_report(store, slots, args.stage, [row["name"] for row in catalog["labels"]])
                output = directory / "exports" / args.stage / "report.json"
                atomic_json(output, report)
                _print({key: report[key] for key in ("stage", "accepted", "expected", "missing", "complete")}
                       | {"report": str(output), "api_usage": report["api_usage"]})
                return 0 if report["complete"] else 2
        except Timeout:
            raise DatasetError("Build is running; request the report after it finishes.") from None
    inspector = TextInspector(settings.model_dir, settings.max_tokens)
    if args.command == "verify":
        result = verify_dataset(settings, catalog, inspector, args.stage)
        _print({key: result[key] for key in ("stage", "ok", "accepted", "expected")}
               | {"error_count": len(result["errors"]),
                  "details": str(state_directory(settings, args.stage) / "exports" / args.stage / "verification.json")})
        return 0 if result["ok"] else 2
    provider = LLMProvider(settings.root, catalog, max_retries=0 if args.command == "smoke" else 3)
    try:
        if args.command == "upgrade":
            _print(upgrade_dataset(settings, catalog, inspector, provider.metadata()))
            return 0
        if args.command == "check":
            _print({"ok": True, "mode": "offline_no_api_calls", "labels": len(catalog["labels"]),
                    "locales": list(settings.locales), "pilot": len(plan_slots(settings, catalog, "pilot")),
                    "full": len(plan_slots(settings, catalog, "full")), "credentials": "present_not_tested",
                    "models": provider.metadata(), "assets": inspector.metadata(),
                    "runtime": asdict(load_runtime(settings.root)),
                    "output_directory": str(settings.output_dir)})
            return 0
        runtime = load_runtime(settings.root, generator_workers=args.generator_workers,
                               judge_workers=args.judge_workers) if args.command == "build" else None
        stage = "smoke" if args.command == "smoke" else args.stage
        report = run_build(settings, catalog, inspector, provider, stage,
                           resume=args.resume if args.command == "build" else False,
                           until_complete=args.command == "build" and not args.bounded, runtime=runtime)
        _print({key: report[key] for key in ("stage", "accepted", "expected", "missing", "complete")}
               | {"outputs": str(state_directory(settings, stage) / "exports" / stage)})
        return 0 if report["complete"] else 2
    finally:
        provider.close()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except DatasetError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Accepted samples are checkpointed; use the same build command with --resume.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, KeyError, OSError):
        # Detailed exceptions can contain data or credential input. Keep stderr safe.
        print("Error: invalid configuration, dataset or filesystem state; check the config and output paths.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
