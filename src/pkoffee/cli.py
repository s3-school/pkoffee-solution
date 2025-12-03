"""Command-line interface for PKoffee analysis."""

import argparse
from enum import StrEnum
from pathlib import Path

from pkoffee.log import LogLevel, init_logging
from pkoffee.productivity_analysis import analyze


class PKoffeCommands(StrEnum):
    """Commands of the pkoffe CLI."""

    ANALYZE = "analyze"


class UnsupportedCommandError(NotImplementedError):
    """Unsupported Command Error."""

    def __init__(self, command: str) -> None:
        super().__init__(f"{command} is not implemented.")


class PKoffeArgParseFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Combine the RawTextHelpFormatter and ArgumentDefaultsHelpFormatter.

    The purpose of this class is to not format description and epilog of the parser (behavior of
    `RawTextHelpFormatter`) while showing defaults for arguments (behavior of `ArgumentDefaultsHelpFormatter`).
    """


def pkoffe_argparser() -> argparse.ArgumentParser:
    """Define the arguments of the PKoffe CLI."""
    parser = argparse.ArgumentParser(
        description="PKoffee - Coffee Productivity Analysis Tool",
        formatter_class=PKoffeArgParseFormatter,
        epilog="""
Examples:
  # Analyze data and create visualization
  pkoffee analyze data.csv --output results.png

  # Show model rankings without plot
  pkoffee analyze data.csv --show-rankings --no-plot

  # Create comparison plot
  pkoffee analyze data.csv --comparison comparison.png
        """,
    )

    parser.add_argument("--log-file", dest="log_file", type=Path, default=None, help="Log file path.")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=LogLevel.from_string,
        choices=tuple(LogLevel),
        default=LogLevel.NOTSET,
        nargs="?",
        const=LogLevel.NOTSET,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        PKoffeCommands.ANALYZE.value,
        help="Analyze coffee productivity data",
        formatter_class=PKoffeArgParseFormatter,
    )
    analyze_parser.add_argument(
        "data-file",
        type=str,
        dest="data_file",
        help="Path to CSV file with 'cups' and 'productivity' columns",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="analysis.png",
        help="Output path for the analysis plot (default: analysis.png)",
    )
    analyze_parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
        help="If supplied, also create a comparison plot with individual model panels at this path",
    )
    analyze_parser.add_argument(
        "--show-rankings",
        action="store_true",
        dest="show_rankings",
        help="Print model rankings to console",
    )
    analyze_parser.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        help="Skip creating the main analysis plot",
    )
    analyze_parser.add_argument(
        "--no-show",
        action="store_true",
        dest="no_show",
        help="Don't display plots (only save to files)",
    )
    analyze_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved figures (default: 150)",
    )
    analyze_parser.add_argument(
        "--y-min",
        type=float,
        dest="y_min",
        help="Minimum y-axis value",
    )
    analyze_parser.add_argument(
        "--y-max",
        type=float,
        dest="y_max",
        help="Maximum y-axis value",
    )
    return parser


def main() -> None:
    """Parse arguments and execute input command."""
    parser = pkoffe_argparser()
    args = parser.parse_args()

    init_logging(args.log_file, args.log_level)

    match args.command:
        case PKoffeCommands.ANALYZE:
            analyze(args)
        case _:
            raise UnsupportedCommandError(args.command)


if __name__ == "__main__":
    main()
