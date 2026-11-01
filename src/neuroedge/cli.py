"""Command-line entry points. Commands do not invent or silently substitute data."""
import argparse
import json
from .config import load_config, environment, project_path, output_path, json_write


def main():
    parser = argparse.ArgumentParser(prog="neuroedge")
    parser.add_argument("command", choices=["env", "prepare", "audit", "run", "report", "deploy", "demo", "verify"])
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    if args.command == "env":
        print(json.dumps(environment(), indent=2))
        return
    cfg = load_config(args.config)
    if args.command in ("prepare", "audit"):
        from .data import prepare
        prepare(cfg)
        sources = []
        for subject in cfg["dataset"]["subjects"]:
            for run in cfg["dataset"]["runs"]:
                path = project_path(cfg, cfg["dataset"]["raw_dir"]) / f"S{subject:03d}/S{subject:03d}R{run:02d}.source.json"
                sources.append({"subject": subject, "run": run, **json.loads(path.read_text())})
        json_write(output_path(cfg) / "acquisition_sources.json", sources)
    elif args.command == "run":
        from .experiment import run
        run(cfg)
    elif args.command == "report":
        from .report import report
        report(cfg)
    elif args.command == "deploy":
        from .deployment import deploy
        deploy(cfg)
    elif args.command == "demo":
        from .demo import build_demo
        build_demo(cfg)
    elif args.command == "verify":
        from .verify import verify
        verify(cfg)


if __name__ == "__main__":
    main()

