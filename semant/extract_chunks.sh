OUTPUT=/mnt/zfs1/data2/2025-05-11_db_benchmarks/lidovky/chunks
#/mnt/matylda1/ikohut/data/smart_digiline/metakat/output_samples.2025-05-13/monograph/*.json
for FILE in ./json/*.json
do
  BASENAME=$(basename "$FILE")
  CHUNK_FILE="${BASENAME%.*}.chunk.jsonl"
  DOC_FILE="${BASENAME%.*}.doc.json"
  if [ -f "${OUTPUT}/$CHUNK_FILE" ]; then
    echo "Chunk file $CHUNK_FILE already exists, skipping."
    continue
  fi

  python prepare_documents.py \
    -i "$FILE" \
    --output-chunk-file "${OUTPUT}/$CHUNK_FILE" \
    --output-doc-file "${OUTPUT}/$DOC_FILE" \
    --page-xml-dir ./xml
done
