# Heretic: Fully automatic censorship removal for language models

You may read up on Heretic [here]

This is a fork of Heretic that supports running the jobs in AML so that you can use larger GPUs.

## Running on Azure ML

Heretic can be run on Azure Machine Learning for cloud-based execution with GPU compute. This is useful for running large-scale abliteration jobs without local hardware constraints.

### Prerequisites

- An Azure subscription with an Azure ML workspace
- GPU compute cluster configured in your workspace (e.g., `Standard_NC6s_v3` or similar)
- Azure CLI installed and authenticated
- Python 3.10+ with Azure ML SDK v2

### Setup

### Configure your Azure ML workspace connection by editing `ml_client_config.json`

```json
{
  "subscription_id": "your-subscription-id",
  "resource_group": "your-resource-group",
  "workspace_name": "your-workspace-name"
}
```

### Install Azure ML dependencies

  ```bash
  pip install azure-ai-ml azure-identity
  ```

### Submitting a Job

Use the `submit_job.py` script to submit abliteration jobs to Azure ML:

```bash
python submit_job.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --n-trials 200 \
  --cluster-name "gpu4-v100" \
  --hf-token "your-huggingface-token"
```

The script will:

- Connect to your Azure ML workspace
- Load the component specification from `component_spec.yaml`
- Submit the job to the specified compute cluster
- Provide a link to monitor job progress in Azure ML Studio

### Component Specification

The Azure ML component is defined in `component_spec.yaml` and supports the following parameters:

- `model`: Hugging Face model ID (required)
- `hf_token`: Hugging Face authentication token (optional, required for gated models)
- `n_trials`: Number of optimization trials (default: 200)
- `n_startup_trials`: Number of random startup trials (default: 60)
- `batch_size`: Batch size for processing, 0 for auto-detection (default: 0)
- `max_batch_size`: Maximum batch size during auto-detection (default: 128)

### Output

The component automatically saves the top 3 optimized models to the output directory, which can be accessed through Azure ML Studio or downloaded using the Azure ML SDK.

If you specify `hf_token`, it will automatically attempt to upload the best one to HuggingFace as well
