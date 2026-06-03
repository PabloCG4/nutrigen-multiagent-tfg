import os
import json
import pandas as pd
from src.vision import config

def build_product_text_mapping(dataset_directory, csv_dump_filepath, output_json_filepath):
    """
    Reads local barcode folders, cross-references them with the offline 
    Open Food Facts CSV using chunking to prevent memory overflow, 
    and generates a JSON dictionary of product names.
    """

    # Creates a list with the barcodes to search
    local_barcodes = []
    for folder_name in os.listdir(dataset_directory):
        folder_path = os.path.join(dataset_directory, folder_name)
        if os.path.isdir(folder_path):
            local_barcodes.append(folder_name)
            
    print(f"Found {len(local_barcodes)} barcode folders locally.")
    local_barcodes_set = set(local_barcodes)
    
    product_mapping = {}
    rows_per_chunk = 100000
    processed_lines_count = 0
    
    try:
        # Load the CSV in manageable chunks of 100k rows
        # Using iterator to process the massive Open Food Facts database efficiently
        csv_chunk_iterator = pd.read_csv(
            csv_dump_filepath, 
            sep='\t', 
            usecols=['code', 'product_name'], 
            dtype=str, 
            chunksize=rows_per_chunk,
            low_memory=False
        )
        
        # Iterate through each chunk and build the mapping for barcodes that exist in our local dataset
        for data_chunk in csv_chunk_iterator:
            # Drop rows missing the barcode or the product name
            valid_dataframe_rows = data_chunk.dropna(subset=['code', 'product_name'])
            
            # Vectorized filtering: keep only rows where the barcode is in our local set
            matched_dataframe_rows = valid_dataframe_rows[valid_dataframe_rows['code'].isin(local_barcodes_set)]
            
            # Iterate only through the matches
            for row_data in matched_dataframe_rows.itertuples(index=False):
                extracted_barcode = str(row_data.code).strip()
                extracted_product_name = str(row_data.product_name).strip()
                
                if extracted_product_name:
                    # Store the mapping in the dictionary. Each row name is the barcode, and the value is the product name.
                    product_mapping[extracted_barcode] = extracted_product_name
            
            processed_lines_count += rows_per_chunk
            # Log progress every 10 chunks (1 million lines)
            if (processed_lines_count / rows_per_chunk) % 10 == 0:
                print(f"Processed {processed_lines_count} lines...")
                
    except Exception as error_message:
        print(f"Failed processing CSV chunk: {error_message}")
        return

    print(f"Saving JSON to: {output_json_filepath}")
    
    output_file_handler = open(output_json_filepath, 'w', encoding='utf-8')
    # As product_mapping is a dictionary, json.dump will serialize it directly to JSON format in the output file.
    json.dump(product_mapping, output_file_handler, ensure_ascii=False, indent=4)
    output_file_handler.close()

if __name__ == "__main__":
    DATASET_DIRECTORY = config.CLASSIFICATION_DATASET_DIR
    
    CSV_DUMP_FILEPATH = os.path.join(config.DATA_DIR, "en.openfoodfacts.org.products.csv")
    OUTPUT_JSON_FILEPATH = os.path.join(config.DATA_DIR, "product_names_mapping.json")
    
    build_product_text_mapping(DATASET_DIRECTORY, CSV_DUMP_FILEPATH, OUTPUT_JSON_FILEPATH)