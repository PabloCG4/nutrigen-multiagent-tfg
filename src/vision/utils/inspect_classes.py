import fiftyone.utils.openimages as fouo

def display_open_images_classes():
    all_classes = fouo.get_classes()
    
    print(f"OPENIMAGES VALID CLASSES ({len(all_classes)})")

    # Sort classes alphabetically
    all_classes.sort()

    for index, class_name in enumerate(all_classes, start=1):
        print(f"{index:3}. {class_name}")

    print("End of class list.")

if __name__ == "__main__":
    display_open_images_classes()