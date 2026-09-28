import lance
import pyarrow as pa
import pandas as pd
import numpy as np

def inspect_lance():
    try:
        ds = lance.dataset("hf://datasets/lance-format/BDD100K-enriched/data/bdd100k.lance", storage_options={})
        print("Total rows:", ds.count_rows())
        
        print("\n--- Inspecting Unique Values ---")
        # We can just fetch the columns we need to find unique values
        df_meta = ds.to_table(columns=["weather", "timeofday"]).to_pandas()
        print("Weather values:", df_meta["weather"].unique())
        print("Time-of-day values:", df_meta["timeofday"].unique())
        
        print("\n--- Inspecting Categories ---")
        df_cats = ds.to_table(columns=["ann_categories"]).to_pandas()
        all_cats = set()
        for cats in df_cats["ann_categories"]:
            if cats is not None:
                for c in cats:
                    all_cats.add(c)
        print("Available classes:", sorted(list(all_cats)))

        print("\n--- Inspecting 5 Sample Rows ---")
        df_sample = ds.to_table(limit=5, columns=["image_id", "width", "height", "ann_categories", "ann_bboxes"]).to_pandas()
        for idx, row in df_sample.iterrows():
            print(f"\nSample {idx+1}:")
            print(f"Image ID: {row['image_id']}")
            print(f"Dimensions: {row['width']}x{row['height']}")
            print(f"Categories: {row['ann_categories']}")
            print(f"BBoxes: {row['ann_bboxes']}")
            
            # verify bbox format
            if len(row['ann_bboxes']) > 0:
                print("First bbox:", row['ann_bboxes'][0])

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    inspect_lance()
