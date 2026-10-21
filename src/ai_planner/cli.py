"""Small offline CLI. Rendering never retrains or simulates."""
import argparse
import json
from pathlib import Path
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for command in ("demo","train"):
        p=commands.add_parser(command)
        p.add_argument("experiment",choices=["analytic","routes"])
        p.add_argument("--config",required=True)
        p.add_argument("--output")
        p.add_argument("--resume")
        p.add_argument("--exact-only",action="store_true")
    p=commands.add_parser("evaluate")
    p.add_argument("--checkpoint",required=True); p.add_argument("--config",required=True); p.add_argument("--output")
    for name in ("render","audit"):
        p=commands.add_parser(name); p.add_argument("--run",required=True)
    args=parser.parse_args()
    if args.command=="render":
        from .visualization import render
        render(Path(args.run))
    elif args.command=="audit":
        root=Path(args.run)
        checks=json.loads((root/"reference_checks.json").read_text())
        finite=json.loads((root/"finite_grid.json").read_text())
        failures=[]
        for row in checks:
            if row["rejection_max_standardized_error"]>5:
                failures.append(f'{row["context"]}: rejection quadrature discrepancy exceeds 5 standard errors')
        def visit(value):
            if isinstance(value,dict):
                if "exact_stationarity_tv" in value and value["exact_stationarity_tv"]>1e-9:
                    failures.append("finite exact MH stationarity failure")
                for v in value.values(): visit(v)
            elif isinstance(value,list):
                for v in value: visit(v)
        visit(finite)
        print(json.dumps({"passed":not failures,"checks":checks,"failures":failures},indent=2))
        if failures: raise SystemExit(1)
    else:
        from .evaluation import run_experiment
        config=yaml.safe_load(Path(args.config).read_text())
        kind=getattr(args,"experiment",None) or config["experiment"]
        output=args.output or config.get("output",f"outputs/{kind}_development")
        path=run_experiment(kind,config,output,checkpoint=getattr(args,"checkpoint",None),
                            resume=getattr(args,"resume",None),exact_only=getattr(args,"exact_only",False))
        print(f"Saved actual run artifacts to {path}")


if __name__=="__main__":
    main()
