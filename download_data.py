from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Sssunset/Earth-Bench",
    repo_type="dataset",
    local_dir="./benchmark/data"
)