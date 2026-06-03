from ultralytics import YOLO



def trigger_dataset_download():

    """
    Loads a base YOLO model to trigger the internal download script 
    defined in custom_sku.yaml.
    """
    print(" Triggering Ultralytics dataset downloader...")
    model = YOLO("yolov8n.pt")
    yaml_filepath = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\data\vision\custom_sku.yaml"

    # Neccesary to trigger the download and export of the dataset, as the .yaml file contains the logic for downloading and preparing the SKU-110K dataset.
    model.train(data=yaml_filepath, epochs=1)

if __name__ == "__main__":
    trigger_dataset_download()