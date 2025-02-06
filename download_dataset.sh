# Usage:
#     bash download_dataset.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     bash download_dataset.sh WaterDrop /tmp/

set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print commands and their arguments as they are executed.

# Check that both parameters are provided.
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 DATASET_NAME OUTPUT_DIR"
  exit 1
fi

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/${DATASET_NAME}"

# Base URL for the dataset.
BASE_URL="https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/${DATASET_NAME}/"

# Create the output directory if it doesn't exist.
mkdir -p "${OUTPUT_DIR}"

# List of files to download.
for file in metadata.json train.tfrecord valid.tfrecord test.tfrecord; do
  wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done
