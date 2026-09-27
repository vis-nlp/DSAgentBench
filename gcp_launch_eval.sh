cd "/home/<your_username>/DSWorld"
USER_HOME="/home/<your_username>/DSWorld"
MINICONDA_PATH="$USER_HOME/miniconda3"
export PATH="$MINICONDA_PATH/bin:$PATH"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"

conda activate osworld
cd OSWorld/

python eval_dataset_gcp.py \
    --model-name 'gpt-5-mini-2025-08-07' \
    --setup-name 'screenshot_a11y_tree' \
    --dataset-name "<your_username>/<dataset_name>" \
    --results-folder "/home/<your_username>/results_gpt5_mini/"
