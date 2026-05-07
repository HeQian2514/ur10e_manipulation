import os
import zarr
import numpy as np
import pandas as pd
import json
from pathlib import Path

def zarr_to_excel(zarr_path, output_excel=None):
    """
    Convert zarr file to Excel format
    
    Args:
    zarr_path: Path to the zarr file or folder
    output_excel: Path to the output Excel file, defaults to strongly-named xlsx sibling file
    """
    # Determine output path
    if output_excel is None:
        output_excel = str(Path(zarr_path).with_suffix('')) + '.xlsx'
    
    print(f"Opening zarr file: {zarr_path}")
    
    # Open zarr file
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"Cannot open zarr file: {e}")
        return
    
    # Use context manager to create Excel writer
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Get and save metadata
        try:
            if hasattr(root, 'attrs'):
                attrs = root.attrs.asdict()
                if attrs:
                    print("Found metadata, saving it to 'metadata' sheet")
                    # Flatten nested dicts to strings
                    flat_attrs = {k: json.dumps(v) if isinstance(v, (dict, list)) else v 
                                for k, v in attrs.items()}
                    pd.DataFrame(flat_attrs.items(), columns=['Key', 'Value']).to_excel(
                        writer, sheet_name='metadata', index=False)
        except Exception as e:
            print(f"Error processing metadata: {e}")
        
        # Process all groups
        for group_name in root:
            # Skip special zarr metadata
            if group_name.startswith('.'):
                continue
                
            print(f"Processing group: {group_name}")
            group = root[group_name]
            
            # Save directly if it is an array
            if isinstance(group, zarr.core.Array):
                save_array_to_excel(group, group_name, writer)
                continue
                
            # Process arrays within the group
            for array_name in group:
                if array_name.startswith('.'):
                    continue
                    
                full_name = f"{group_name}/{array_name}"
                print(f"  Processing array: {full_name}")
                
                try:
                    array = group[array_name]
                    save_array_to_excel(array, full_name, writer)
                except Exception as e:
                    print(f"  Error processing array {full_name}: {e}")
    
    # Context manager used, no need to explicitly save
    print(f"Saved Excel file: {output_excel}")
    print("Conversion completed!")

def save_array_to_excel(array, name, writer):
    """Save zarr array to Excel worksheet"""
    # Replace unsupported characters in Excel sheet name
    sheet_name = name.replace('/', '_')[:31]  # Excel sheet name limit is 31 chars
    
    # Different processing based on array dimensions
    shape = array.shape
    
    if len(shape) == 1 or (len(shape) == 2 and shape[1] < 100):
        # 1D array or small 2D array - save directly
        df = pd.DataFrame(array[:])
        df.to_excel(writer, sheet_name=sheet_name)
    
    elif len(shape) == 2:
        # Large 2D array - save summary and samples
        n_samples = min(10, shape[0])
        summary_df = pd.DataFrame({
            "Array Info": [
                f"Shape: {shape}",
                f"Data Type: {array.dtype}",
                f"Total Elements: {array.size}",
                "First 10 rows of sample data shown below"
            ]
        })
        summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Save sample data into same sheet, shifting down a few rows
        sample_data = array[:n_samples]
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_excel(writer, sheet_name=sheet_name, startrow=len(summary_df)+2)
    
    else:
        # High dimensional array - save info and first element
        info = [
            f"Shape: {shape}",
            f"Data Type: {array.dtype}",
            f"Total Elements: {array.size}",
            f"Dimensions: {len(shape)}",
            "Data is too complex to fully display in Excel"
        ]
        
        # Save first element as sample
        sample = array[0]
        if isinstance(sample, np.ndarray):
            if sample.size < 1000:
                sample_str = str(sample)
            else:
                sample_str = "Sample is too large to display"
        else:
            sample_str = str(sample)
            
        info.append(f"First element sample: {sample_str}")
        
        pd.DataFrame({"Array Info": info}).to_excel(
            writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert zarr file to Excel format')
    parser.add_argument('--zarr_path', default='data/example/learn.zarr', help='Path to zarr file or folder')
    parser.add_argument('--output', '-o', help='Path to output Excel file (optional)')
    
    args = parser.parse_args()
    zarr_to_excel(args.zarr_path, args.output)

    '''python convert_zarr_to_excel.py /path/to/your/output/replay-buffer.zarr'''