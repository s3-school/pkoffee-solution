import json
from pathlib import Path


def escape_latex(s):
    return s.replace("_", r"\_")


def write_line(f, s="", indent=0):
    indent = " " * indent
    f.write(f"{indent}{s}\n")


def write_model(f, model):
    colspec = "{" + " ".join(["l"] + ["r"] * len(model["params"])) + "}"
    write_line(f, r"\begin{tblr}{")
    write_line(f, f"colspec = {colspec},", indent=2)
    write_line(f, "}")

    param_names = [escape_latex(p) for p in model["params"]]
    write_line(f, " & ".join(["  Model", *param_names]) + r" \\")
    f.write("  " + model["name"])
    for param in model["params"].values():
        f.write(" & ")
        f.write(f"{param:.4g}")

    write_line(f, r"\\")
    write_line(f, r"\end{tblr}")
    write_line(f)



with Path("build/model.json").open() as f:
    models = json.load(f)

with Path("build/model.tex").open("w") as f:
    for model in models["Models"]:
        write_model(f, model)
