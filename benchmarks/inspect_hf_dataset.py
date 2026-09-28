import datasets

def inspect_dataset():
    dataset_id = "lance-format/BDD100K-enriched"
    try:
        builder = datasets.load_dataset_builder(dataset_id)
        
        print("Dataset:", dataset_id)
        print("Description:", builder.info.description)
        print("Features:", builder.info.features)
        print("Splits:", builder.info.splits)
        
        print("Configs:", builder.builder_configs)
    except Exception as e:
        print("Error with builder:", e)

    print("\n--- Inspecting via streaming ---")
    try:
        dataset = datasets.load_dataset(dataset_id, split="train", streaming=True)
        it = iter(dataset)
        sample = next(it)
        print("Keys in sample:", sample.keys())
        
        # print non-image fields
        for k, v in sample.items():
            if k == "image" or str(type(v)).find("JpegImageFile") != -1 or str(type(v)).find("PngImageFile") != -1:
                print(f"{k}: <Image data>")
            else:
                print(f"{k}: {v}")
                
        print("\n--- Second sample ---")
        sample2 = next(it)
        for k, v in sample2.items():
            if k == "image" or str(type(v)).find("JpegImageFile") != -1 or str(type(v)).find("PngImageFile") != -1:
                print(f"{k}: <Image data>")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print("Error streaming:", e)

if __name__ == "__main__":
    inspect_dataset()
