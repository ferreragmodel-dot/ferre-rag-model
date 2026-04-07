import json
import re
from pathlib import Path


INPUT_PATH = Path("generated_image_metadata.jsonl")
OUTPUT_PATH = Path("generated_image_metadata_fixed_paths.jsonl")


def is_missing(value):
    """Treat None, empty string, and literal 'None' as missing."""
    return value is None or str(value).strip() == "" or str(value).strip().lower() == "none"


def parse_from_source_path(source_path):
    """
    Parse season_path and year_path from source_path.

    Examples:
    - 'ALTA MODA 1987 SS/Womenswear/Fashion show photos/62219.jpg'
        -> season_path='SS1987', year_path='1987'
    - 'ALTA MODA 1988-89 FW/Womenswear/...'
        -> season_path='FW1988-1989', year_path='1988'
    """
    if not source_path:
        return None, None

    path_str = str(source_path).strip()

    # Case 1: 1987 SS / 1987 FW
    m1 = re.search(r'(\d{4})\s*(SS|FW)', path_str, re.IGNORECASE)
    if m1:
        year = m1.group(1)
        season_code = m1.group(2).upper()
        season_path = f"{season_code}{year}"
        year_path = year
        return season_path, year_path

    # Case 2: 1988-89 FW / 1988-89 SS
    m2 = re.search(r'(\d{4})-(\d{2})\s*(SS|FW)', path_str, re.IGNORECASE)
    if m2:
        start_year = m2.group(1)
        end_suffix = m2.group(2)
        season_code = m2.group(3).upper()

        century_prefix = start_year[:2]
        end_year = century_prefix + end_suffix

        season_path = f"{season_code}{start_year}-{end_year}"
        year_path = start_year
        return season_path, year_path

    return None, None


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    records = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

    total_records = len(records)
    initially_missing = []
    fixed_count = 0
    unresolved_after_fix = []

    # First pass: identify and repair only season_path/year_path
    for idx, rec in enumerate(records):
        season_missing = is_missing(rec.get("season_path"))
        year_missing = is_missing(rec.get("year_path"))

        if season_missing or year_missing:
            initially_missing.append(idx)

            parsed_season_path, parsed_year_path = parse_from_source_path(rec.get("source_path"))

            if season_missing and parsed_season_path is not None:
                rec["season_path"] = parsed_season_path

            if year_missing and parsed_year_path is not None:
                rec["year_path"] = parsed_year_path

            # Count as fixed only if both are no longer missing
            if not is_missing(rec.get("season_path")) and not is_missing(rec.get("year_path")):
                fixed_count += 1

    # Second pass: global check
    for idx, rec in enumerate(records):
        if is_missing(rec.get("season_path")) or is_missing(rec.get("year_path")):
            unresolved_after_fix.append({
                "index": idx,
                "source_path": rec.get("source_path"),
                "season_path": rec.get("season_path"),
                "year_path": rec.get("year_path"),
            })

    # Write output
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print summary
    print(f"Total records: {total_records}")
    print(f"Initially missing season_path/year_path: {len(initially_missing)}")
    print(f"Successfully fixed: {fixed_count}")
    print(f"Remaining missing after fix: {len(unresolved_after_fix)}")
    print(f"Output written to: {OUTPUT_PATH}")

    if unresolved_after_fix:
        print("\nRemaining unresolved entries:")
        for item in unresolved_after_fix[:20]:
            print(item)


if __name__ == "__main__":
    main()