import os
import json
import pandas as pd
from src.vision import config


# Open Food Facts CSV export: nutriments per 100g (same logical keys as API nutriments *_100g).
# See https://world.openfoodfacts.org/data — field names may vary slightly between dump versions.
_MACRO_COLUMN_ALIASES = {
    "energy_kcal": ("energy-kcal_100g", "energy_kcal_100g"),
    "proteins": ("proteins_100g",),
    "fat": ("fat_100g",),
    "carbohydrates": ("carbohydrates_100g", "carbohydrate_100g"),
}


def _pick_column(available: set[str], aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name in available:
            return name
    return None


def _resolve_off_macro_columns(csv_dump_filepath: str) -> dict[str, str | None]:
    header_df = pd.read_csv(csv_dump_filepath, sep="\t", nrows=0, low_memory=False)
    available = set(header_df.columns)
    resolved: dict[str, str | None] = {}
    for logical, aliases in _MACRO_COLUMN_ALIASES.items():
        resolved[logical] = _pick_column(available, aliases)
    return resolved


def _to_float_per_100g(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def build_product_macros_mapping(dataset_directory, csv_dump_filepath, output_json_filepath):
    """
    Reads local barcode folders, cross-references them with the offline
    Open Food Facts CSV using chunking to prevent memory overflow,
    and generates a JSON dictionary: barcode -> product_name + macros per 100g.
    """

    local_barcodes = []
    for folder_name in os.listdir(dataset_directory):
        folder_path = os.path.join(dataset_directory, folder_name)
        if os.path.isdir(folder_path):
            local_barcodes.append(folder_name)

    print(f"Found {len(local_barcodes)} barcode folders locally.")
    local_barcodes_set = set(local_barcodes)

    macro_cols = _resolve_off_macro_columns(csv_dump_filepath)
    missing_macro = [k for k, v in macro_cols.items() if v is None]
    if missing_macro:
        print(
            "Warning: could not find CSV columns for: "
            + ", ".join(missing_macro)
            + ". Those fields will be null when absent in other columns."
        )

    usecols_list = ["code", "product_name"]
    for _logical, colname in macro_cols.items():
        if colname is not None and colname not in usecols_list:
            usecols_list.append(colname)

    product_mapping = {}
    rows_per_chunk = 100000
    processed_lines_count = 0

    def macro_from_row(row: pd.Series, logical: str) -> float | None:
        col = macro_cols.get(logical)
        if not col:
            return None
        return _to_float_per_100g(row.get(col))

    try:
        csv_chunk_iterator = pd.read_csv(
            csv_dump_filepath,
            sep="\t",
            usecols=usecols_list,
            dtype=str,
            chunksize=rows_per_chunk,
            low_memory=False,
        )

        for data_chunk in csv_chunk_iterator:
            valid_dataframe_rows = data_chunk.dropna(subset=["code", "product_name"])
            matched_dataframe_rows = valid_dataframe_rows[
                valid_dataframe_rows["code"].isin(local_barcodes_set)
            ]

            for _, row in matched_dataframe_rows.iterrows():
                extracted_barcode = str(row["code"]).strip()
                extracted_product_name = str(row["product_name"]).strip()

                if not extracted_product_name:
                    continue

                product_mapping[extracted_barcode] = {
                    "product_name": extracted_product_name,
                    "kcal_per_100g": macro_from_row(row, "energy_kcal"),
                    "protein_g_per_100g": macro_from_row(row, "proteins"),
                    "fat_g_per_100g": macro_from_row(row, "fat"),
                    "carbs_g_per_100g": macro_from_row(row, "carbohydrates"),
                }

            processed_lines_count += rows_per_chunk
            if (processed_lines_count / rows_per_chunk) % 10 == 0:
                print(f"Processed {processed_lines_count} lines...")

    except Exception as error_message:
        print(f"Failed processing CSV chunk: {error_message}")
        return

    print(f"Saving JSON to: {output_json_filepath}")

    output_file_handler = open(output_json_filepath, "w", encoding="utf-8")
    json.dump(product_mapping, output_file_handler, ensure_ascii=False, indent=4)
    output_file_handler.close()


if __name__ == "__main__":
    DATASET_DIRECTORY = config.CLASSIFICATION_DATASET_DIR

    CSV_DUMP_FILEPATH = os.path.join(config.DATA_DIR, "en.openfoodfacts.org.products.csv")
    OUTPUT_JSON_FILEPATH = os.path.join(config.DATA_DIR, "product_test_database_macros.json")

    build_product_macros_mapping(DATASET_DIRECTORY, CSV_DUMP_FILEPATH, OUTPUT_JSON_FILEPATH)
