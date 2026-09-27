USER_HOME="${HOME:-/home/$USER}"
MINICONDA_PATH="$USER_HOME/miniconda3"
export PATH="$MINICONDA_PATH/bin:$PATH"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"

conda init
conda activate dsworld
cd "$USER_HOME/DSWorld" 2>/dev/null || cd "$(dirname "$0")"

python run_data_science.py \
    --model "bytedance-research/UI-TARS-7B-DPO" \
    --observation_type 'screenshot_a11y_tree' \
    --provider_name docker \
    --path_to_vm "" \
    --domain data_science

# Example: Forward cluster vLLM port
# ssh -L 8000:localhost:8000 <user>@<cluster_host>

# Example: Serve UI-TARS with vLLM
# vllm serve "bytedance-research/UI-TARS-7B-DPO" \
#   --dtype "bfloat16" \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --api-key "abc123"