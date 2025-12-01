"""
Command-line interface for PKoffee analysis.
"""

import argparse
import sys

from pkoffee.data import load_data
from pkoffee.models import fit_all_models
from pkoffee.visualization import (
    create_analysis_plot,
    create_comparison_plot,
    print_model_rankings,
)


def main() -> int:
    """
    Main entry point for the PKoffee CLI.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="PKoffee - Coffee Productivity Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze coffee productivity data",
    )
    analyze_parser.add_argument(
        "data_file",
        type=str,
        help="Path to CSV file with 'cups' and 'productivity' columns",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="analysis.png",
        help="Output path for the analysis plot (default: analysis.png)",
    )
    analyze_parser.add_argument(
        "--comparison",
        type=str,
        help="Also create a comparison plot with individual model panels",
    )
    analyze_parser.add_argument(
        "--show-rankings",
        action="store_true",
        help="Print model rankings to console",
    )
    analyze_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip creating the main analysis plot",
    )
    analyze_parser.add_argument(
        "--no-show",
        action="store_true",
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
        help="Minimum y-axis value",
    )
    analyze_parser.add_argument(
        "--y-max",
        type=float,
        help="Maximum y-axis value",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "analyze":
        return analyze_command(args)
    
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    """
    Execute the analyze command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code.
    """
    try:
        # Load data
        print(f"Loading data from: {args.data_file}")
        data = load_data(args.data_file)
        print(f"Loaded {len(data)} data points")
        
        # Fit models
        print("\nFitting models...")
        fitted_models = fit_all_models(data)
        
        if not fitted_models:
            print("Error: No models successfully fitted to the data")
            return 1
        
        print(f"Successfully fitted {len(fitted_models)} models")
        
        # Print rankings if requested
        if args.show_rankings:
            print_model_rankings(fitted_models)
        
        # Determine y-axis limits
        y_limits = None
        if args.y_min is not None or args.y_max is not None:
            y_min = args.y_min if args.y_min is not None else data["productivity"].min()
            y_max = args.y_max if args.y_max is not None else data["productivity"].max()
            y_limits = (y_min, y_max)
        
        # Create main analysis plot
        if not args.no_plot:
            print(f"\nCreating analysis plot: {args.output}")
            create_analysis_plot(
                data,
                fitted_models,
                output_path=args.output,
                dpi=args.dpi,
                y_limits=y_limits,
                show=not args.no_show,
            )
        
        # Create comparison plot if requested
        if args.comparison:
            print(f"\nCreating comparison plot: {args.comparison}")
            create_comparison_plot(
                data,
                fitted_models,
                output_path=args.comparison,
                dpi=args.dpi,
            )
        
        print("\n✓ Analysis complete!")
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
