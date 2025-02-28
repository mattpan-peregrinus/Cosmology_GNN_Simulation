#!/usr/bin/env bash
# download_dataset.sh
# Usage: ./download_dataset.sh WaterDropSmall ./temp/datasets

DATASET_NAME="$1"
OUTPUT_DIR="$2/$DATASET_NAME"

BASE_URL="https://storage.googleapis.com/cs224w_course_project_dataset/${DATASET_NAME}"

mkdir -p "$OUTPUT_DIR"

echo "Downloading metadata.json..."
wget -O "$OUTPUT_DIR/metadata.json" "${BASE_URL}/metadata.json"

echo "Downloading data splits..."
for split in test train valid
do
  for suffix in offset.json particle_type.dat position.dat
  do
    DATA_PATH="${OUTPUT_DIR}/${split}_${suffix}"
    CLOUD_PATH="${BASE_URL}/${split}_${suffix}"
    wget -O "$DATA_PATH" "$CLOUD_PATH"
  done
done

echo "Finished downloading dataset ${DATASET_NAME} to ${OUTPUT_DIR}"
