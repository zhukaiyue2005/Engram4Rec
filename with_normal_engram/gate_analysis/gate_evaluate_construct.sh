# The number of processes can only be one for inference
export CUDA_VISIBLE_DEVICES="2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
DATA_FILE="$PARENT_DIR/../data/Industrial_and_Scientific_dataset/test.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/engram_results_test"
CHECKPOINT_PATH="../output_sft_layer=6,13,20_tbs=128_lora=3e-4_engram=3e-4/checkpoint-1988"
MASTER_PORT=31311

cd "$SCRIPT_DIR"

torchrun --nproc_per_node 1 --master_port="$MASTER_PORT" \
    gate_evaluate_construct.py \
    --batch_size 8 \
    --base_model "$BASE_MODEL" \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --resume_from_checkpoint "$CHECKPOINT_PATH" \
    --engram_layer_ids auto \
    > eval_gate.log 2>&1
