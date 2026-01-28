import json
from pathlib import Path


def escape_latex(s):
    return s.replace("_", r"\_")


def write_line(f, s="", indent=0):
    indent = " " * indent
    f.write(f"{indent}{s}\n")


def write_all_models(f, models_list):
    write_line(f, r"\begin{table}")
    write_line(f, r"\centering")
    write_line(f, r"\caption{Best-fit parameters for all models.}")
    write_line(f, r"\vspace{0.5\baselineskip}")

    # Calculate number of columns: Model name + parameters for each model
    num_models = len(models_list)
    col_spec = "l" + " r" * num_models
    write_line(f, rf"\begin{{tabular}}{{@{{\extracolsep\fill}} {col_spec} @{{\extracolsep\fill}}}}")
    write_line(f, r"\toprule", indent=2)

    # Header row with model names
    header = "Parameter"
    for model in models_list:
        header += f" & {model['name']}"
    header += r" \\"
    write_line(f, header, indent=2)
    write_line(f, r"\midrule", indent=2)

    # Collect all parameter names across all models
    all_params = set()
    for model in models_list:
        all_params.update(model["params"].keys())
    all_params = sorted(all_params)

    # Write parameter rows
    for param in all_params:
        param_escaped = escape_latex(param)
        row = param_escaped
        for model in models_list:
            if param in model["params"]:
                row += f" & {model['params'][param]:.4g}"
            else:
                row += " & ---"
        row += r" \\"
        write_line(f, row, indent=2)

    write_line(f, r"\bottomrule", indent=2)
    write_line(f, r"\end{tabular}")
    write_line(f, r"\end{table}")
    write_line(f)


with Path("build/model.json").open() as f:
    models = json.load(f)

with Path("build/model.tex").open("w") as f:
    write_all_models(f, models["Models"])
