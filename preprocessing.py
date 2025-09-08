import os
import shutil
import cv2
import pydicom
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------
# Step 0: create processed dataset folders
folders = [
    "datasets/processed/images/train",
    "datasets/processed/images/val",
    "datasets/processed/labels/train",
    "datasets/processed/labels/val"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

# -------------------
# Step 1: RSNA preprocessing

def process_rsna(rsna_dir="datasets/rsna/", csv_file="datasets/rsna/stage_2_train_labels.csv"):

    print("Processing RSNA dataset...")

    # Read DICOM files
    all_files = [f for f in os.listdir(rsna_dir) if f.endswith(".dcm")]
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # Convert DICOM → JPG
    for dataset, file_list in zip(["train","val"], [train_files, val_files]):
        for f in file_list:
            ds = pydicom.dcmread(os.path.join(rsna_dir, f))
            img = ds.pixel_array
            img = cv2.convertScaleAbs(img, alpha=(255.0/img.max()))
            out_path = os.path.join(f"datasets/processed/images/{dataset}", f.replace(".dcm",".jpg"))
            cv2.imwrite(out_path, img)

    # Convert CSV bounding boxes to YOLO format
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        img_file = row["patientId"] + ".jpg"
        if row["Target"] == 1:  # pneumonia
            # figure out train/val folder
            if img_file in train_files:
                label_path = f"datasets/processed/labels/train/{img_file.replace('.jpg','.txt')}"
                img_path = f"datasets/processed/images/train/{img_file}"
            elif img_file in val_files:
                label_path = f"datasets/processed/labels/val/{img_file.replace('.jpg','.txt')}"
                img_path = f"datasets/processed/images/val/{img_file}"
            else:
                continue

            img = cv2.imread(img_path)
            h, w, _ = img.shape
            x_c = (row['x'] + row['width']/2)/w
            y_c = (row['y'] + row['height']/2)/h
            bw = row['width']/w
            bh = row['height']/h

            with open(label_path, "a") as f:
                f.write(f"0 {x_c} {y_c} {bw} {bh}\n")  # class 0 = pneumonia

    print("RSNA dataset processed successfully!")

# -------------------
# Step 2: MURA preprocessing

def process_mura(mura_dir="datasets/mura/"):

    print("Processing MURA dataset...")

    # Collect all images
    all_images = []
    for root, _, files in os.walk(mura_dir):
        for f in files:
            if f.endswith(".png") or f.endswith(".jpg"):
                all_images.append(os.path.join(root, f))

    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

    def copy_and_label(img_list, img_dest, label_dest):
        for img_path in img_list:
            fname = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(img_dest, fname))
            label_path = os.path.join(label_dest, fname.replace(".png",".txt").replace(".jpg",".txt"))
            if "positive" in img_path.lower():  # fracture
                with open(label_path, "w") as f:
                    f.write("1 0.5 0.5 1.0 1.0\n")  # full image bbox
            else:  # normal
                open(label_path,"w").close()  # empty file

    copy_and_label(train_imgs, "datasets/processed/images/train", "datasets/processed/labels/train")
    copy_and_label(val_imgs, "datasets/processed/images/val", "datasets/processed/labels/val")

    print("MURA dataset processed successfully!")

# -------------------
if __name__ == "__main__":
    process_rsna()
    process_mura()
    print("All preprocessing done! YOLO dataset is ready.")
