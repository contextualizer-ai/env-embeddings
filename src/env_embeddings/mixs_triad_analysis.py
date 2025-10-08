"""MIxS environmental triad ontology term analysis for biosample metadata.

This module analyzes ontology terms (primarily ENVO) used in the MIxS environmental
context triad fields across GOLD, NCBI, and NMDC data sources.

The MIxS environmental triad consists of:
- env_broad_scale: Broadest environmental context (typically biome)
- env_local_scale: Local environmental context
- env_medium: Environmental material/medium

Analysis includes:
1. Frequency analysis to identify rare terms for ML training
2. Parent class analysis showing ontological ancestry

Examples:
    >>> # Analyze all sources
    >>> analyze_triad_terms("data/gold_file.tsv", "gold", min_pct=0.1)

    >>> # Analyze specific column
    >>> analyze_column_parents("data/ncbi_file.tsv", "env_medium", "ncbi")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import pandas as pd
import typer
from rich.console import Console  # type: ignore[import-untyped, import-not-found]
from rich.table import Table  # type: ignore[import-untyped, import-not-found]
from oaklib import get_adapter  # type: ignore[import-untyped, import-not-found]
from oaklib.interfaces import OntologyInterface  # type: ignore[import-untyped, import-not-found]
import tabulate  # type: ignore[import-untyped]  # noqa: F401  # Used by pandas.DataFrame.to_markdown()

app = typer.Typer(help="Analyze ontology terms in MIxS environmental triad fields")
console = Console()

# MIxS environmental triad column names
MIXS_TRIAD_COLUMNS = ["env_broad_scale", "env_local_scale", "env_medium"]

# Expected root parents (ENVO-specific)
EXPECTED_PARENTS = {
    "env_medium": "ENVO:00010483",  # environmental material
    "env_broad_scale": "ENVO:00000428",  # biome
}

# Default exclusion list for high-level ontology classes
# These are typically uninformative upper ontology terms from BFO
DEFAULT_EXCLUDED_PARENTS = [
    # BFO (Basic Formal Ontology) - very high level classes
    "BFO:0000001",  # entity
    "BFO:0000002",  # continuant
    "BFO:0000003",  # occurrent
    "BFO:0000004",  # independent continuant
    "BFO:0000016",  # disposition
    "BFO:0000017",  # realizable entity
    "BFO:0000020",  # specifically dependent continuant
    "BFO:0000024",  # fiat object part
    "BFO:0000031",  # generically dependent continuant
    "BFO:0000040",  # material entity
]


def load_data(file_path: Path) -> pd.DataFrame:
    """Load TSV or CSV data file.

    Args:
        file_path: Path to the data file

    Returns:
        DataFrame with the data

    Examples:
        >>> df = load_data(Path("data/gold_file.tsv"))
        >>> "env_broad_scale" in df.columns
        True
    """
    if file_path.suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    return pd.read_csv(file_path)


def get_ontology_adapter() -> OntologyInterface:
    """Get oaklib adapter for ENVO ontology.

    Note: Currently uses ENVO as the primary ontology for MIxS triad terms,
    but could be extended to support other ontologies.

    Returns:
        OntologyInterface for ENVO

    Examples:
        >>> adapter = get_ontology_adapter()
        >>> adapter is not None
        True
    """
    return get_adapter("sqlite:obo:envo")


def calculate_cumulative_coverage(
    term_counts: pd.Series,
) -> Dict[str, Union[int, float]]:
    """Calculate what percentage of terms account for X% of occurrences.

    Args:
        term_counts: Series with term counts

    Returns:
        Dictionary with percentile coverage statistics

    Examples:
        >>> counts = pd.Series([100, 50, 25, 10, 5])
        >>> stats = calculate_cumulative_coverage(counts)
        >>> 'terms_for_50pct' in stats
        True
    """
    total = term_counts.sum()
    cumsum = term_counts.cumsum()
    cumsum_pct = cumsum / total * 100

    stats: Dict[str, Union[int, float]] = {}
    for target_pct in [50, 80, 95]:
        # Find how many terms needed to reach target% of occurrences
        if (cumsum_pct >= target_pct).any():
            # Get position where threshold is crossed
            idx = (cumsum_pct >= target_pct).argmax()
            terms_needed = int(idx) + 1
        else:
            terms_needed = len(term_counts)

        pct_of_terms = terms_needed / len(term_counts) * 100
        stats[f"terms_for_{target_pct}pct"] = int(terms_needed)
        stats[f"pct_terms_for_{target_pct}pct"] = float(round(pct_of_terms, 2))

    return stats


def analyze_term_frequency(
    df: pd.DataFrame,
    column: str,
    adapter: OntologyInterface,
    min_pct: float = 0.01,
    min_absolute: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """Analyze frequency of ontology terms in a MIxS triad column.

    Args:
        df: DataFrame with ontology terms in MIxS triad columns
        column: Column name to analyze (env_broad_scale, env_local_scale, or env_medium)
        adapter: oaklib adapter for the ontology
        min_pct: Minimum percentage threshold (e.g., 0.1 for 0.1%)
        min_absolute: Minimum absolute count (overrides percentage if higher)

    Returns:
        Tuple of (DataFrame with term frequencies, statistics dict)

    Examples:
        >>> adapter = get_ontology_adapter()
        >>> df = pd.DataFrame({"env_medium": ["ENVO:00001998"] * 100 + ["ENVO:00002001"] * 50})
        >>> result, stats = analyze_term_frequency(df, "env_medium", adapter, min_pct=1.0)
        >>> len(result) >= 1
        True
    """
    # Get term counts
    term_counts = df[column].value_counts()
    total_records = len(df)

    # Calculate effective minimum based on percentage
    min_count_from_pct = int(total_records * (min_pct / 100))
    effective_min = max(min_count_from_pct, min_absolute)

    # Filter by minimum frequency
    filtered = term_counts[term_counts >= effective_min]

    # Create result dataframe
    result = pd.DataFrame(
        {
            "term": filtered.index,
            "count": filtered.values,
            "percentage": (filtered.values.astype(float) / total_records * 100).round(
                3
            ),
        }
    )

    # Add labels
    result["label"] = result["term"].apply(lambda x: adapter.label(x) or "")

    # Calculate statistics (convert numpy types to Python types for JSON serialization)
    stats = {
        "total_records": int(total_records),
        "total_unique_terms": int(len(term_counts)),
        "terms_after_filter": int(len(filtered)),
        "effective_min_count": int(effective_min),
        "min_pct_used": float(min_pct),
        # Rare term counts
        "terms_below_0.01pct": int((term_counts < int(total_records * 0.0001)).sum()),
        "terms_below_0.1pct": int((term_counts < int(total_records * 0.001)).sum()),
        "terms_below_0.5pct": int((term_counts < int(total_records * 0.005)).sum()),
        "terms_below_1pct": int((term_counts < int(total_records * 0.01)).sum()),
    }

    # Add cumulative coverage stats
    coverage = calculate_cumulative_coverage(term_counts)
    stats.update(coverage)

    return result, stats


def get_all_parents(term: str, adapter: OntologyInterface) -> List[str]:
    """Get all parent terms (ancestors) for a given ontology term.

    Args:
        term: Ontology term ID (e.g., "ENVO:00001998")
        adapter: oaklib adapter for the ontology

    Returns:
        List of unique parent term IDs (including the term itself)

    Note:
        Uses reflexive=True so counts include the term itself plus all subclasses.
        Deduplicates results because terms with multiple parents appear multiple times.

    Examples:
        >>> adapter = get_ontology_adapter()
        >>> parents = get_all_parents("ENVO:00001998", adapter)
        >>> isinstance(parents, list)
        True
    """
    try:
        # Get all ancestors including term itself (reflexive=True for correct counting)
        # Note: terms with multiple parents appear multiple times, so we deduplicate
        ancestors = list(
            adapter.ancestors(term, predicates=["rdfs:subClassOf"], reflexive=True)
        )
        # Deduplicate - terms with multiple inheritance paths appear once per path
        return list(set(ancestors))
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not get parents for {term}: {e}[/yellow]"
        )
        return []


def analyze_parent_classes(
    df: pd.DataFrame,
    column: str,
    adapter: OntologyInterface,
    exclude_parents: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Analyze parent classes for all ontology terms in a MIxS triad column.

    Args:
        df: DataFrame with ontology terms in MIxS triad columns
        column: Column name to analyze (env_broad_scale, env_local_scale, or env_medium)
        adapter: oaklib adapter for the ontology
        exclude_parents: List of parent term IDs to exclude from results (default: DEFAULT_EXCLUDED_PARENTS)

    Returns:
        DataFrame with parent term frequencies sorted by count

    Examples:
        >>> adapter = get_ontology_adapter()
        >>> df = pd.DataFrame({"env_medium": ["ENVO:00001998", "ENVO:00001998"]})
        >>> result = analyze_parent_classes(df, "env_medium", adapter)
        >>> "parent_term" in result.columns
        True
    """
    if exclude_parents is None:
        exclude_parents = DEFAULT_EXCLUDED_PARENTS

    # Get unique terms
    unique_terms = df[column].dropna().unique()

    # Collect all parents with their term counts
    parent_counter: Counter[str] = Counter()

    console.print(
        f"[cyan]Analyzing {len(unique_terms)} unique terms in {column}...[/cyan]"
    )

    for term in unique_terms:
        # Get count for this term
        term_count = (df[column] == term).sum()

        # Get all parents
        parents = get_all_parents(term, adapter)

        # Add each parent with the term's count (excluding filtered ones)
        for parent in parents:
            if parent not in exclude_parents:
                parent_counter[parent] += term_count

    # Create result dataframe
    result = pd.DataFrame(
        {
            "parent_term": list(parent_counter.keys()),
            "count": list(parent_counter.values()),
        }
    )

    # Add labels
    result["label"] = result["parent_term"].apply(lambda x: adapter.label(x) or "")

    # Sort by count
    result = result.sort_values("count", ascending=False)

    return result


def print_frequency_table(results: Dict[str, Tuple[pd.DataFrame, Dict]], source: str):
    """Print frequency analysis results as rich tables.

    Args:
        results: Dictionary mapping column names to (frequency DataFrame, stats dict) tuples
        source: Data source name (e.g., "GOLD", "NCBI")
    """
    for column, (df, stats) in results.items():
        console.print(f"\n[bold cyan]═══ {source.upper()} - {column} ═══[/bold cyan]")
        console.print(
            f"[dim]Total records: {stats['total_records']:,} | "
            f"Unique terms: {stats['total_unique_terms']:,} | "
            f"Min threshold: {stats['min_pct_used']:.2f}% ({stats['effective_min_count']} records)[/dim]\n"
        )

        # Top terms table
        console.print("[bold]Top 20 Most Frequent Terms[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Term", style="cyan")
        table.add_column("Label", style="blue")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")

        top_df = df.head(20)
        for _, row in top_df.iterrows():
            table.add_row(
                str(row["term"]),
                str(row["label"]),
                f"{row['count']:,}",
                f"{row['percentage']:.3f}%",
            )

        console.print(table)

        # Bottom terms table (rarest that passed filter)
        if len(df) > 20:
            console.print(
                f"\n[bold yellow]Bottom 10 Rarest Terms (that passed {stats['min_pct_used']:.2f}% threshold)[/bold yellow]"
            )
            bottom_table = Table(show_header=True, header_style="bold yellow")
            bottom_table.add_column("Term", style="cyan")
            bottom_table.add_column("Label", style="blue")
            bottom_table.add_column("Count", justify="right", style="yellow")
            bottom_table.add_column("Percentage", justify="right", style="red")

            bottom_df = df.tail(10)
            for _, row in bottom_df.iterrows():
                bottom_table.add_row(
                    str(row["term"]),
                    str(row["label"]),
                    f"{row['count']:,}",
                    f"{row['percentage']:.3f}%",
                )

            console.print(bottom_table)

        # Print enhanced summary statistics
        console.print("\n[bold]Class Imbalance Analysis[/bold]")
        console.print(
            f"[cyan]Terms accounting for 50% of occurrences:[/cyan] "
            f"{stats['terms_for_50pct']} ({stats['pct_terms_for_50pct']:.1f}% of unique terms)"
        )
        console.print(
            f"[cyan]Terms accounting for 80% of occurrences:[/cyan] "
            f"{stats['terms_for_80pct']} ({stats['pct_terms_for_80pct']:.1f}% of unique terms)"
        )
        console.print(
            f"[cyan]Terms accounting for 95% of occurrences:[/cyan] "
            f"{stats['terms_for_95pct']} ({stats['pct_terms_for_95pct']:.1f}% of unique terms)"
        )

        # Add imbalance severity rating
        imbalance_ratio = stats["pct_terms_for_50pct"]
        if imbalance_ratio < 5:
            severity = "SEVERE"
            color = "red"
            recommendation = "Consider aggressive filtering (min_pct ≥ 1.0%) or parent class grouping"
        elif imbalance_ratio < 15:
            severity = "HIGH"
            color = "yellow"
            recommendation = (
                "Recommended min_pct ≥ 0.5% or use parent classes for rare terms"
            )
        elif imbalance_ratio < 30:
            severity = "MODERATE"
            color = "blue"
            recommendation = "Recommended min_pct ≥ 0.1%"
        else:
            severity = "LOW"
            color = "green"
            recommendation = "Dataset relatively balanced, min_pct ≥ 0.01% should work"

        console.print(f"\n[bold {color}]Imbalance Severity: {severity}[/bold {color}]")
        console.print(f"[dim]{recommendation}[/dim]")

        console.print("\n[bold]Rare Terms (ML Training Concerns)[/bold]")
        console.print(
            f"[yellow]Terms < 0.01% ({int(stats['total_records'] * 0.0001):,} records):[/yellow] "
            f"{stats['terms_below_0.01pct']:,}"
        )
        console.print(
            f"[yellow]Terms < 0.1% ({int(stats['total_records'] * 0.001):,} records):[/yellow] "
            f"{stats['terms_below_0.1pct']:,}"
        )
        console.print(
            f"[yellow]Terms < 0.5% ({int(stats['total_records'] * 0.005):,} records):[/yellow] "
            f"{stats['terms_below_0.5pct']:,}"
        )
        console.print(
            f"[yellow]Terms < 1.0% ({int(stats['total_records'] * 0.01):,} records):[/yellow] "
            f"{stats['terms_below_1pct']:,}"
        )

        # Add data quality warnings
        rare_term_pct = stats["terms_below_0.1pct"] / stats["total_unique_terms"] * 100
        if rare_term_pct > 50:
            console.print("\n[bold red]⚠ DATA QUALITY WARNING:[/bold red]")
            console.print(
                f"[red]{rare_term_pct:.1f}% of terms have <0.1% frequency (likely insufficient for ML)[/red]"
            )
            console.print(
                "[dim]Consider: (1) parent class grouping, (2) increasing min_pct threshold, (3) data augmentation[/dim]"
            )
        elif rare_term_pct > 25:
            console.print("\n[bold yellow]⚠ Note:[/bold yellow]")
            console.print(
                f"[yellow]{rare_term_pct:.1f}% of terms have <0.1% frequency[/yellow]"
            )
            console.print(
                "[dim]Many rare classes may need special handling (grouping or exclusion)[/dim]"
            )


def print_parent_table(results: Dict[str, pd.DataFrame], source: str):
    """Print parent class analysis results as rich tables.

    Args:
        results: Dictionary mapping column names to parent DataFrames
        source: Data source name (e.g., "GOLD", "NCBI")
    """
    for column, df in results.items():
        console.print(
            f"\n[bold cyan]═══ {source.upper()} - {column} Parent Classes ═══[/bold cyan]\n"
        )

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parent Term", style="cyan")
        table.add_column("Label", style="blue")
        table.add_column("Count", justify="right", style="green")

        # Show top 30 parents
        top_df = df.head(30)
        for _, row in top_df.iterrows():
            # Highlight expected parents
            term_style = (
                "bold green"
                if row["parent_term"] in EXPECTED_PARENTS.values()
                else "cyan"
            )
            table.add_row(
                f"[{term_style}]{row['parent_term']}[/{term_style}]",
                str(row["label"]),
                str(row["count"]),
            )

        console.print(table)

        # Check for expected parents
        if column in EXPECTED_PARENTS:
            expected = EXPECTED_PARENTS[column]
            if expected in df["parent_term"].values:
                rank = int((df["parent_term"] == expected).idxmax()) + 1
                count = df[df["parent_term"] == expected]["count"].values[0]
                console.print(
                    f"\n[green]✓ Expected parent {expected} found at rank {rank} with count {count}[/green]"
                )
            else:
                console.print(f"\n[red]✗ Expected parent {expected} not found[/red]")


def generate_markdown_report(
    all_stats: Dict[str, Dict[str, Tuple[pd.DataFrame, Dict]]],
    parent_results: Dict[str, Dict[str, pd.DataFrame]],
    output_file: Path,
):
    """Generate markdown report of the analysis.

    Args:
        all_stats: Nested dict of {source: {column: (df, stats)}}
        parent_results: Nested dict of {source: {column: parent_df}}
        output_file: Path to write markdown file
    """
    lines = []
    lines.append("# MIxS Environmental Triad Analysis Report\n")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Cross-dataset comparison
    lines.append("\n## Cross-Dataset Comparison\n")

    for column in MIXS_TRIAD_COLUMNS:
        lines.append(f"\n### {column}\n")

        # Collect metrics
        metrics_data = {}
        for source in ["gold", "ncbi", "nmdc"]:
            if source in all_stats and column in all_stats[source]:
                _, stats = all_stats[source][column]
                metrics_data[source] = stats

        if not metrics_data:
            lines.append(f"*No data available for {column}*\n")
            continue

        # Create comparison table
        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Records",
                    "Unique Terms",
                    "Terms < 0.1%",
                    "Terms for 50% coverage",
                ],
                "GOLD": [
                    f"{metrics_data.get('gold', {}).get('total_records', 0):,}",
                    f"{metrics_data.get('gold', {}).get('total_unique_terms', 0):,}",
                    f"{metrics_data.get('gold', {}).get('terms_below_0.1pct', 0):,}",
                    f"{metrics_data.get('gold', {}).get('terms_for_50pct', 0)} ({metrics_data.get('gold', {}).get('pct_terms_for_50pct', 0):.1f}%)",
                ],
                "NCBI": [
                    f"{metrics_data.get('ncbi', {}).get('total_records', 0):,}",
                    f"{metrics_data.get('ncbi', {}).get('total_unique_terms', 0):,}",
                    f"{metrics_data.get('ncbi', {}).get('terms_below_0.1pct', 0):,}",
                    f"{metrics_data.get('ncbi', {}).get('terms_for_50pct', 0)} ({metrics_data.get('ncbi', {}).get('pct_terms_for_50pct', 0):.1f}%)",
                ],
                "NMDC": [
                    f"{metrics_data.get('nmdc', {}).get('total_records', 0):,}",
                    f"{metrics_data.get('nmdc', {}).get('total_unique_terms', 0):,}",
                    f"{metrics_data.get('nmdc', {}).get('terms_below_0.1pct', 0):,}",
                    f"{metrics_data.get('nmdc', {}).get('terms_for_50pct', 0)} ({metrics_data.get('nmdc', {}).get('pct_terms_for_50pct', 0):.1f}%)",
                ],
            }
        )

        lines.append(comparison_df.to_markdown(index=False))
        lines.append("\n")

    # Detailed frequency analysis per source
    lines.append("\n## Detailed Frequency Analysis\n")

    for source in ["gold", "ncbi", "nmdc"]:
        if source not in all_stats:
            continue

        lines.append(f"\n### {source.upper()}\n")

        for column, (df, stats) in all_stats[source].items():
            lines.append(f"\n#### {column}\n")
            lines.append(f"- Total records: {stats['total_records']:,}\n")
            lines.append(f"- Unique terms: {stats['total_unique_terms']:,}\n")
            lines.append(f"- Terms after filter: {stats['terms_after_filter']:,}\n")

            # Class imbalance
            imbalance_ratio = stats["pct_terms_for_50pct"]
            if imbalance_ratio < 5:
                severity = "🔴 SEVERE"
            elif imbalance_ratio < 15:
                severity = "🟡 HIGH"
            elif imbalance_ratio < 30:
                severity = "🔵 MODERATE"
            else:
                severity = "🟢 LOW"

            lines.append(f"- **Imbalance Severity:** {severity}\n")
            lines.append(
                f"- Terms for 50% coverage: {stats['terms_for_50pct']} ({stats['pct_terms_for_50pct']:.1f}%)\n"
            )
            lines.append(
                f"- Terms for 80% coverage: {stats['terms_for_80pct']} ({stats['pct_terms_for_80pct']:.1f}%)\n"
            )
            lines.append(
                f"- Terms for 95% coverage: {stats['terms_for_95pct']} ({stats['pct_terms_for_95pct']:.1f}%)\n"
            )

            # Top 20 terms
            lines.append("\n**Top 20 Most Frequent Terms:**\n\n")
            top_20 = df.head(20)[["term", "label", "count", "percentage"]]
            lines.append(top_20.to_markdown(index=False))
            lines.append("\n")

            # Rare term warnings
            rare_term_pct = (
                stats["terms_below_0.1pct"] / stats["total_unique_terms"] * 100
            )
            if rare_term_pct > 50:
                lines.append(
                    f"\n⚠️ **DATA QUALITY WARNING:** {rare_term_pct:.1f}% of terms have <0.1% frequency\n"
                )
                lines.append(
                    "- Consider: parent class grouping, increasing min_pct threshold, or data augmentation\n"
                )
            elif rare_term_pct > 25:
                lines.append(
                    f"\n⚠️ **Note:** {rare_term_pct:.1f}% of terms have <0.1% frequency\n"
                )
                lines.append("- Many rare classes may need special handling\n")

    # Parent class analysis
    if parent_results:
        lines.append("\n## Parent Class Analysis\n")

        for source in ["gold", "ncbi", "nmdc"]:
            if source not in parent_results:
                continue

            lines.append(f"\n### {source.upper()} Parent Classes\n")

            for column, parent_df in parent_results[source].items():
                lines.append(f"\n#### {column}\n")

                # Top 30 parents
                top_30 = parent_df.head(30)[["parent_term", "label", "count"]]
                lines.append(top_30.to_markdown(index=False))
                lines.append("\n")

                # Expected parent check
                if column in EXPECTED_PARENTS:
                    expected = EXPECTED_PARENTS[column]
                    if expected in parent_df["parent_term"].values:
                        rank = int((parent_df["parent_term"] == expected).idxmax()) + 1
                        count = parent_df[parent_df["parent_term"] == expected][
                            "count"
                        ].values[0]
                        lines.append(
                            f"✓ Expected parent `{expected}` found at rank {rank} with count {count:,}\n"
                        )
                    else:
                        lines.append(f"✗ Expected parent `{expected}` not found\n")

    # ML recommendations
    lines.append("\n## ML Strategy Recommendations\n")
    lines.append(
        "\n1. **Start with NMDC** if available (typically cleanest metadata)\n"
    )
    lines.append("2. **Use parent class grouping** for terms below your threshold\n")
    lines.append(
        "3. **Consider separate models** per dataset if vocabularies differ significantly\n"
    )
    lines.append(
        "4. **For combined training**, normalize term usage across datasets first\n"
    )
    lines.append(
        "5. **Monitor class imbalance** - severe imbalance may require min_pct ≥ 1.0%\n"
    )

    # Write to file
    output_file.write_text("\n".join(lines))
    console.print(f"[green]✓ Markdown report saved to {output_file}[/green]")


def print_dataset_comparison(
    all_stats: Dict[str, Dict[str, Tuple[pd.DataFrame, Dict]]],
):
    """Print cross-dataset comparison for ML decision-making.

    Args:
        all_stats: Nested dict of {source: {column: (df, stats)}}
    """
    console.print(f"\n[bold magenta]{'═' * 80}[/bold magenta]")
    console.print("[bold magenta]  CROSS-DATASET COMPARISON FOR ML[/bold magenta]")
    console.print(f"[bold magenta]{'═' * 80}[/bold magenta]\n")

    for column in MIXS_TRIAD_COLUMNS:
        console.print(f"\n[bold cyan]━━━ {column} ━━━[/bold cyan]")

        comparison_table = Table(
            show_header=True, header_style="bold blue", title=f"{column} Statistics"
        )
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("GOLD", justify="right", style="yellow")
        comparison_table.add_column("NCBI", justify="right", style="green")
        comparison_table.add_column("NMDC", justify="right", style="magenta")

        # Collect data for this column across all sources
        metrics_data = {}
        for source in ["gold", "ncbi", "nmdc"]:
            if source in all_stats and column in all_stats[source]:
                _, stats = all_stats[source][column]
                metrics_data[source] = stats

        if not metrics_data:
            console.print(f"[dim]No data for {column}[/dim]")
            continue

        # Add rows
        comparison_table.add_row(
            "Total Records",
            f"{metrics_data.get('gold', {}).get('total_records', 0):,}",
            f"{metrics_data.get('ncbi', {}).get('total_records', 0):,}",
            f"{metrics_data.get('nmdc', {}).get('total_records', 0):,}",
        )

        comparison_table.add_row(
            "Unique Terms",
            f"{metrics_data.get('gold', {}).get('total_unique_terms', 0):,}",
            f"{metrics_data.get('ncbi', {}).get('total_unique_terms', 0):,}",
            f"{metrics_data.get('nmdc', {}).get('total_unique_terms', 0):,}",
        )

        comparison_table.add_row(
            "Terms < 0.1%",
            f"{metrics_data.get('gold', {}).get('terms_below_0.1pct', 0):,}",
            f"{metrics_data.get('ncbi', {}).get('terms_below_0.1pct', 0):,}",
            f"{metrics_data.get('nmdc', {}).get('terms_below_0.1pct', 0):,}",
        )

        comparison_table.add_row(
            "Terms for 50% coverage",
            f"{metrics_data.get('gold', {}).get('terms_for_50pct', 0)} "
            f"({metrics_data.get('gold', {}).get('pct_terms_for_50pct', 0):.1f}%)",
            f"{metrics_data.get('ncbi', {}).get('terms_for_50pct', 0)} "
            f"({metrics_data.get('ncbi', {}).get('pct_terms_for_50pct', 0):.1f}%)",
            f"{metrics_data.get('nmdc', {}).get('terms_for_50pct', 0)} "
            f"({metrics_data.get('nmdc', {}).get('pct_terms_for_50pct', 0):.1f}%)",
        )

        console.print(comparison_table)

        # Add ML recommendations per column
        console.print(f"\n[bold]ML Recommendations for {column}:[/bold]")

        # Find dataset with best characteristics for ML
        best_for_ml = None
        best_score = float("inf")

        for source, stats in metrics_data.items():
            # Lower score = better for ML (fewer rare terms, lower imbalance)
            rare_pct = (
                stats.get("terms_below_0.1pct", 0)
                / max(stats.get("total_unique_terms", 1), 1)
                * 100
            )
            imbalance = stats.get("pct_terms_for_50pct", 100)
            score = rare_pct + (
                100 - imbalance
            )  # Penalize rare terms and high imbalance

            if score < best_score:
                best_score = score
                best_for_ml = source

        if best_for_ml:
            console.print(
                f"[green]✓ {best_for_ml.upper()} has best ML characteristics for {column}[/green]"
            )

        # Identify common challenges
        all_have_high_imbalance = all(
            stats.get("pct_terms_for_50pct", 100) < 10
            for stats in metrics_data.values()
        )
        if all_have_high_imbalance:
            console.print(
                f"[yellow]⚠ All datasets show severe imbalance in {column}[/yellow]"
            )
            console.print(
                "[dim]  → Consider parent class grouping to reduce class count[/dim]"
            )

        # Check if term vocabularies are similar across datasets
        term_counts = [
            stats.get("total_unique_terms", 0) for stats in metrics_data.values()
        ]
        max_diff = max(term_counts) - min(term_counts)
        if max_diff > max(term_counts) * 0.5:  # >50% difference
            console.print(
                f"[yellow]⚠ Large vocabulary difference across datasets ({min(term_counts)} to {max(term_counts)} terms)[/yellow]"
            )
            console.print(
                "[dim]  → Cross-dataset model may need vocabulary alignment[/dim]"
            )

    console.print("\n[bold]Overall ML Strategy Recommendations:[/bold]")
    console.print(
        "[cyan]1. Start with NMDC if available (typically cleanest metadata)[/cyan]"
    )
    console.print(
        "[cyan]2. Use parent class grouping for terms below your threshold[/cyan]"
    )
    console.print(
        "[cyan]3. Consider separate models per dataset if vocabularies differ significantly[/cyan]"
    )
    console.print(
        "[cyan]4. For combined training, normalize term usage across datasets first[/cyan]"
    )


@app.command()
def frequency(
    file_path: Path = typer.Option(
        ..., "--file", "-f", help="Path to data file (TSV or CSV)"
    ),
    source: str = typer.Option(
        ..., "--source", "-s", help="Data source name (gold, ncbi, nmdc)"
    ),
    min_pct: float = typer.Option(
        0.1, "--min-pct", "-p", help="Minimum percentage threshold (e.g., 0.1 for 0.1%)"
    ),
    min_absolute: int = typer.Option(
        0, "--min-absolute", "-a", help="Minimum absolute count (optional override)"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
):
    """Analyze frequency of ontology terms in MIxS environmental triad columns.

    Shows which terms occur too infrequently for ML training.
    Uses percentage-based thresholds by default.
    """
    console.print(f"[bold]Loading data from {file_path}...[/bold]")
    df = load_data(file_path)

    console.print(f"[green]Loaded {len(df):,} records[/green]")
    console.print("[cyan]Loading ENVO ontology...[/cyan]")

    adapter = get_ontology_adapter()

    # Analyze each MIxS triad column
    results = {}
    for column in MIXS_TRIAD_COLUMNS:
        if column in df.columns:
            console.print(f"\n[cyan]Analyzing {column}...[/cyan]")
            results[column] = analyze_term_frequency(
                df, column, adapter, min_pct, min_absolute
            )

    # Print results
    print_frequency_table(results, source)

    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for column, (result_df, stats) in results.items():
            output_file = output_dir / f"{source}_{column}_frequency.csv"
            result_df.to_csv(output_file, index=False)

            # Save stats as separate file
            stats_file = output_dir / f"{source}_{column}_stats.json"
            import json

            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

            console.print(f"\n[green]Saved {column} results to {output_file}[/green]")
            console.print(f"[green]Saved {column} stats to {stats_file}[/green]")


@app.command()
def parents(
    file_path: Path = typer.Option(
        ..., "--file", "-f", help="Path to data file (TSV or CSV)"
    ),
    source: str = typer.Option(
        ..., "--source", "-s", help="Data source name (gold, ncbi, nmdc)"
    ),
    column: str = typer.Option(
        None,
        "--column",
        "-c",
        help="Specific MIxS triad column to analyze (default: all)",
    ),
    exclude_terms: str = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Comma-separated list of parent terms to exclude (e.g., 'BFO:0000001,BFO:0000002')",
    ),
    no_default_exclusions: bool = typer.Option(
        False, "--no-default-exclusions", help="Disable default BFO exclusions"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
):
    """Analyze parent classes for MIxS triad ontology terms using oaklib.

    Shows the ontological ancestry of terms, sorted by frequency.
    By default, excludes high-level BFO terms. Use --no-default-exclusions to show all.
    """
    console.print(f"[bold]Loading data from {file_path}...[/bold]")
    df = load_data(file_path)

    console.print(f"[green]Loaded {len(df)} records[/green]")
    console.print("[cyan]Loading ENVO ontology...[/cyan]")

    adapter = get_ontology_adapter()

    # Build exclusion list
    if no_default_exclusions:
        exclude_parents = []
    else:
        exclude_parents = DEFAULT_EXCLUDED_PARENTS.copy()

    if exclude_terms:
        # Add user-specified exclusions
        user_exclusions = [t.strip() for t in exclude_terms.split(",")]
        exclude_parents.extend(user_exclusions)

    if exclude_parents:
        console.print(
            f"[dim]Excluding {len(exclude_parents)} high-level parent terms[/dim]"
        )

    # Determine which MIxS triad columns to analyze
    columns_to_analyze = [column] if column else MIXS_TRIAD_COLUMNS
    columns_to_analyze = [c for c in columns_to_analyze if c in df.columns]

    # Analyze each column
    results = {}
    for col in columns_to_analyze:
        console.print(f"\n[bold cyan]═══ Analyzing {col} ═══[/bold cyan]")
        results[col] = analyze_parent_classes(df, col, adapter, exclude_parents)

    # Print results
    print_parent_table(results, source)

    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for col, result_df in results.items():
            output_file = output_dir / f"{source}_{col}_parents.csv"
            result_df.to_csv(output_file, index=False)
            console.print(
                f"\n[green]Saved {col} parent analysis to {output_file}[/green]"
            )


@app.command()
def analyze_all(
    gold_file: Path = typer.Option(..., "--gold", help="Path to GOLD data file"),
    ncbi_file: Path = typer.Option(..., "--ncbi", help="Path to NCBI data file"),
    nmdc_file: Path = typer.Option(..., "--nmdc", help="Path to NMDC data file"),
    min_pct: float = typer.Option(
        0.1, "--min-pct", "-p", help="Minimum percentage threshold (e.g., 0.1 for 0.1%)"
    ),
    min_absolute: int = typer.Option(
        0, "--min-absolute", "-a", help="Minimum absolute count (optional override)"
    ),
    exclude_terms: str = typer.Option(
        None, "--exclude", "-e", help="Comma-separated list of parent terms to exclude"
    ),
    no_default_exclusions: bool = typer.Option(
        False, "--no-default-exclusions", help="Disable default BFO exclusions"
    ),
    output_dir: Path = typer.Option(
        "results/mixs_triad_analysis", "--output", "-o", help="Output directory"
    ),
):
    """Analyze all three data sources in one command.

    Runs both frequency and parent analysis for GOLD, NCBI, and NMDC.
    Uses percentage-based thresholds by default.
    By default, excludes high-level BFO terms from parent analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("gold", gold_file),
        ("ncbi", ncbi_file),
        ("nmdc", nmdc_file),
    ]

    console.print("[bold]Loading ENVO ontology...[/bold]")
    adapter = get_ontology_adapter()

    # Build exclusion list for parent analysis
    if no_default_exclusions:
        exclude_parents = []
    else:
        exclude_parents = DEFAULT_EXCLUDED_PARENTS.copy()

    if exclude_terms:
        user_exclusions = [t.strip() for t in exclude_terms.split(",")]
        exclude_parents.extend(user_exclusions)

    if exclude_parents:
        console.print(
            f"[dim]Excluding {len(exclude_parents)} high-level parent terms from parent analysis[/dim]"
        )

    # Collect all statistics for cross-dataset comparison
    all_stats = {}
    all_parent_results = {}

    for source_name, file_path in sources:
        console.print(f"\n[bold magenta]{'═' * 60}[/bold magenta]")
        console.print(f"[bold magenta]  {source_name.upper()}[/bold magenta]")
        console.print(f"[bold magenta]{'═' * 60}[/bold magenta]\n")

        df = load_data(file_path)
        console.print(f"[green]Loaded {len(df):,} records[/green]\n")

        # Frequency analysis
        console.print("[bold cyan]FREQUENCY ANALYSIS[/bold cyan]")
        freq_results = {}
        for column in MIXS_TRIAD_COLUMNS:
            if column in df.columns:
                freq_results[column] = analyze_term_frequency(
                    df, column, adapter, min_pct, min_absolute
                )

        print_frequency_table(freq_results, source_name)

        # Store stats for comparison
        all_stats[source_name] = freq_results

        # Save frequency results
        import json

        for column, (result_df, stats) in freq_results.items():
            output_file = output_dir / f"{source_name}_{column}_frequency.csv"
            result_df.to_csv(output_file, index=False)

            # Save stats
            stats_file = output_dir / f"{source_name}_{column}_stats.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

        # Parent analysis
        console.print("\n[bold cyan]PARENT CLASS ANALYSIS[/bold cyan]")
        parent_results = {}
        for column in MIXS_TRIAD_COLUMNS:
            if column in df.columns:
                parent_results[column] = analyze_parent_classes(
                    df, column, adapter, exclude_parents
                )

        print_parent_table(parent_results, source_name)

        # Store parent results for markdown report
        all_parent_results[source_name] = parent_results

        # Save parent results
        for column, result_df in parent_results.items():
            output_file = output_dir / f"{source_name}_{column}_parents.csv"
            result_df.to_csv(output_file, index=False)

    # Print cross-dataset comparison summary
    print_dataset_comparison(all_stats)

    # Generate markdown report
    markdown_file = output_dir / "analysis_report.md"
    generate_markdown_report(all_stats, all_parent_results, markdown_file)

    console.print(f"\n[bold green]✓ All results saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
