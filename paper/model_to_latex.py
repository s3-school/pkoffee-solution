import json
from pathlib import Path


def escape_latex(s):
    return s.replace("_", r"\_")


def write_line(f, s="", indent=0):
    indent = " " * indent
    f.write(f"{indent}{s}\n")


def write_model(f, model):
    write_line(f, r"\begin{table}")
    caption = f"Best-fit parameters of the {model['name']} model."
    write_line(f, r"\centering")
    write_line(f, rf"\caption{{{caption}}}")
    write_line(f, r"\vspace{0.5\baselineskip}")
    write_line(f, r"\begin{tblr}{")
    write_line(f, "colspec = {l r},", indent=2)
    write_line(f, "}")
    write_line(f, r"\toprule", indent=2)
    write_line(f, r"Name & Value \\", indent=2)
    write_line(f, r"\midrule", indent=2)

    for name, value in model["params"].items():
        name = escape_latex(name)
        write_line(f, rf"{name} & {value:.4g} \\", indent=2)

    write_line(f, r"\bottomrule", indent=2)
    write_line(f, r"\end{tblr}")
    write_line(f, r"\end{table}")
    write_line(f)



with Path("build/model.json").open() as f:
    models = json.load(f)

with Path("build/model.tex").open("w") as f:
    for model in models["Models"]:
        write_model(f, model)
