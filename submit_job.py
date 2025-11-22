#!/usr/bin/env python3
"""
Submit a lucidity job to Azure ML using the component specification.

Usage:
    python submit_job.py --model "meta-llama/Llama-2-7b-hf" --n-trials 200

This script will:
1. Connect to your Azure ML workspace
2. Load the component specification from component_spec.yaml
3. Submit a job using the component to the specified compute cluster
4. Monitor the job progress
"""

import argparse
import logging
import os
from typing import Dict
from azure.ai.ml import MLClient, load_component, dsl
from azure.ai.ml.entities import UserIdentityConfiguration, ManagedIdentityConfiguration
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit a lucidity job to Azure ML using the component specification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
            python submit_job.py --model "meta-llama/Llama-2-7b-hf" --n-trials 200
            python submit_job.py --model "mistralai/Mistral-7B-v0.1" --n-trials 100 --stream
        """
    )
    parser.add_argument('--cluster_name', type=str, default="gpu4-v100", help="Name of the cluster to execute the pipeline on")
    parser.add_argument('--stream', type=bool, default=False, help='Stream the job')
    parser.add_argument('--hf-token', type=str, help='HuggingFace token')
    parser.add_argument("--model", type=str, required=True, help="Model name or path (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--n-trials", type=int, default=200, help="Number of optimization trials to run (default: 200)")
    parser.add_argument("--n-startup-trials", type=int, default=60, help="Number of random startup trials (default: 60)")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for processing (0 = auto, default: 0)")
    parser.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch size to try during auto-detection (default: 256)")

    args = parser.parse_args()

    # Log all arguments
    for k, v in args.__dict__.items():
        logging.info(f'{k}: {v}')
    
    return args

def create_pipeline_job(component_inputs: Dict) -> object:
    @dsl.pipeline(display_name = f"Lucidity pipeline")
    def export_optimize_wrapper(hf_token:str, model:str, n_trials:int, n_startup_trials:int, batch_size:int, max_batch_size:int):
        run_lucidity = load_component(source=os.path.join(os.path.dirname(__file__), "component_spec.yaml"))
        run_step = run_lucidity(
            hf_token=hf_token,
            model=model,
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            batch_size=batch_size,
            max_batch_size=max_batch_size)
        return { "raw_results": run_step.outputs.model_outputs }
    return export_optimize_wrapper(
        model=component_inputs['model'],
        hf_token=component_inputs['hf_token'],
        n_trials=component_inputs['n_trials'],
        n_startup_trials=component_inputs['n_startup_trials'],
        batch_size=component_inputs['batch_size'],
        max_batch_size=component_inputs['max_batch_size'])
   
def configure_logging() -> None:
    log_handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log_handlers.append(console_handler)
    logging.basicConfig(
        level=logging._nameToLevel["INFO"],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=log_handlers
        )
    # Suppress noisy logs from azure, urllib3, msrest, etc.
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("msrest.serialization").setLevel(logging.ERROR)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)

   
def get_credential() -> TokenCredential:
    log_key = "get_credential()"
    credential = None
    try:
        url = "https://management.azure.com/.default"
        logging.info(f"{log_key}: Attempting to get credential using default credential")
        credential = DefaultAzureCredential()
        logging.info(f"{log_key}: Attempting to get token for {url}")
        credential.get_token(url)
    except Exception as e:
        logging.error(f"{log_key}: Failed with exception {e}")
    return credential

def get_ml_client(credential: TokenCredential, cluster_name: str) -> MLClient:
    log_key = "get_ml_client()"
    logging.info(f"{log_key}: Getting ML Client for cluster {cluster_name} and retrieving compute")
    ml_client = MLClient.from_config(credential=credential, file_name="ml_client_config.json")
    ml_client.compute.get(cluster_name)
    return ml_client

def submit_job(
    ml_client: MLClient,
    pipeline_job: object,
    cluster_name: str,
    experiment_name: str,
    stream_job: bool = True,
    tags: dict = None
) -> object:
    pipeline_job.settings.default_compute = cluster_name
    pipeline_job.identity = ManagedIdentityConfiguration()
    pipeline_job = ml_client.jobs.create_or_update(
        job=pipeline_job,
        experiment_name=experiment_name,
        tags=tags
    )

    logging.info(f"Pipeline job Name: {pipeline_job.name}")
    logging.info(f"Experiment Name: {experiment_name}")
    logging.info(f"Job link: {pipeline_job.studio_url}")

    if stream_job:
        ml_client.jobs.stream(pipeline_job.name)

    return pipeline_job


if __name__ == "__main__" :
    configure_logging()
    args = parse_arguments()
    if args is None:
        logging.error("Failed to parse arguments.")
        exit()

    cred = get_credential()
    if cred is None:
        raise RuntimeError("Failed to obtain credentials")
    ml_client = get_ml_client(cred, args.cluster_name)

    component_inputs = {
        "hf_token": args.hf_token,
        "model": args.model,
        "n_trials": args.n_trials,
        "n_startup_trials": args.n_startup_trials,
        "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size,
    }
    
    pipeline_job = create_pipeline_job(component_inputs)
    
    job = submit_job(
        ml_client=ml_client,
        pipeline_job=pipeline_job,
        cluster_name=args.cluster_name,
        experiment_name=f'Lucidity',
        stream_job=args.stream,
        tags={ "notes": args.model }
    )
    
    # Extract just the job name (after the last '/')
    job_identifier = job.id.split('/')[-1]
    logging.info(f"Job ID: {job_identifier}")
