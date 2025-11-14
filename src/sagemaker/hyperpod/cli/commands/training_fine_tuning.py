from jinja2 import Template
import boto3
from datetime import datetime, timezone
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import yaml
import json
import os
import sys
import click
from pathlib import Path
from sagemaker.hyperpod.cli.init_utils import load_dynamic_schema
from sagemaker.hyperpod.common.utils import handle_exception
from sagemaker.hyperpod.cli.type_handler_utils import is_undefined_value
from sagemaker.hyperpod.cli.fine_tuning_utils import (
    _fetch_recipe_from_hub, _download_s3_content, _download_s3_json,
    _validate_and_convert_value, _collect_all_parameters_interactively,
    _submit_k8s_resources, _render_k8s_template, _get_sagemaker_client,
    _get_s3_client, _get_k8s_custom_client, _validate_dynamic_template,
    _generate_dynamic_config_yaml, _update_config_field
)
import shutil
from sagemaker.hyperpod.common.telemetry.constants import Feature
from sagemaker.hyperpod.common.telemetry.telemetry_logging import _hyperpod_telemetry_emitter
from sagemaker.hyperpod.common.cli_decorators import handle_cli_exceptions


def _init_fine_tuning_job(directory: str, model_name: str, technique: str, instance_type: str) -> bool:
    """Initialize fine-tuning job configuration."""
    try:
        sagemaker_client = _get_sagemaker_client()
        s3_client = _get_s3_client()
        
        # Fetch and validate recipe
        matching_recipe = _fetch_recipe_from_hub(sagemaker_client, model_name, technique, instance_type)
        
        override_params_uri = matching_recipe.get('HpEksOverrideParamsS3Uri')
        k8s_template_uri = matching_recipe.get('HpEksPayloadTemplateS3Uri')
        
        if not override_params_uri or not k8s_template_uri:
            click.secho("❌ Missing S3 URIs in recipe", fg="red")
            return False
        
        # Create directory
        dir_path = Path(directory).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Download and save override params
        override_data = _download_s3_json(s3_client, override_params_uri)
        with open(dir_path / '.override_spec.json', 'w') as f:
            json.dump(override_data, f, indent=2)
        
        # Create config.yaml
        _generate_dynamic_config_yaml(dir_path, "fine-tuning-job", model_name=model_name, technique=technique, instance_type=instance_type)
        
        # Download and save k8s template
        k8s_content = _download_s3_content(s3_client, k8s_template_uri)
        with open(dir_path / 'k8s.jinja', 'w') as f:
            f.write(k8s_content)
        
        return True
        
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")
        return False

def _configure_dynamic_template(ctx, option, value, dir_path):
    """Handle configure for dynamic templates (fine-tuning)"""
    config_path = dir_path / "config.yaml"
    spec_path = dir_path / ".override_spec.json"
    
    if not spec_path.exists():
        click.secho(f"❌ .override_spec.json not found", fg="red")
        ctx.exit(1)
    
    # Load spec
    spec = load_dynamic_schema(dir_path)
    
    # Check if user provided --option flags (only those explicitly provided, not defaults)
    provided_options = {}
    for param_name, param_value in ctx.params.items():
        if param_name not in ['option', 'value', 'model_config']:
            # Check if this parameter was actually provided by the user (not a default)
            param_source = ctx.get_parameter_source(param_name)
            if param_source and param_source.name == 'COMMANDLINE' and param_value is not None:
                # Convert back to original key format
                original_key = param_name.replace('-', '_')
                if original_key in spec:
                    provided_options[original_key] = param_value
    
    # If --option flags were used, process them
    if provided_options:
        for key, value in provided_options.items():
            _update_config_field(config_path, spec, key, value)
        
        click.secho(f"✔️  config.yaml updated successfully.", fg="green")
        return
    
    # If no arguments, show help like --help does
    click.echo(ctx.get_help())
    ctx.exit(0)
    
    # Validate option exists
    if option not in spec:
        click.secho(f"❌ Unknown option: {option}", fg="red")
        click.echo(f"\nRun 'hyp configure' to see available options")
        ctx.exit(1)
    
    if value is None:
        click.secho(f"❌ This field is required. Please provide a value for option: {option}", fg="red")
        ctx.exit(1)
    
    # Validate and convert value using extracted utility
    try:
        converted_value = _validate_and_convert_value(str(value), spec[option])
    except ValueError as e:
        click.secho(f"❌ {e}", fg="red")
        ctx.exit(1)
    
    # Update config.yaml using extracted utility
    _update_config_field(config_path, spec, option, converted_value)
    click.secho(f"✅ Successfully set {option} = {converted_value}", fg="green")


def _create_dynamic_template(dir_path: Path, config_data: dict):
    """Handle create for dynamic templates (fine-tuning)"""
    try:
        # Validate config first
        _validate_dynamic_template(dir_path)
        click.secho("✔️ Configuration validated successfully", fg="green")
        
        k8s_template_file = dir_path / 'k8s.jinja'
        if not k8s_template_file.exists():
            raise FileNotFoundError("k8s.jinja template not found")
        
        # Read and render template
        template_content = k8s_template_file.read_text()
        rendered = _render_k8s_template(template_content, config_data)
        
        # Create run directory
        run_root = dir_path / 'run'
        run_root.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        out_dir = run_root / timestamp
        out_dir.mkdir()
        
        # Save files
        shutil.copy(dir_path / 'config.yaml', out_dir / 'config.yaml')
        (out_dir / 'k8s.yaml').write_text(rendered)
        
        relative_out_dir = Path("run") / timestamp
        click.secho(f"✔️ Files written to {relative_out_dir}", fg="green")
        
        # Submit to Kubernetes
        custom_api = _get_k8s_custom_client()
        _submit_k8s_resources(custom_api, rendered)
        
        click.secho("✔️ Successfully submitted to Kubernetes", fg="green")
                
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"❌ {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        # Use existing handle_exception for Kubernetes errors
        try:
            resource_name = config_data.get('name', 'unknown')
            handle_exception(e, resource_name, 'default')
        except Exception as handled_e:
            click.secho(f"❌ {handled_e}", fg="red")
        sys.exit(1)


@click.command("fine-tuning-job")
@click.option("--model-name", help="Model name from SageMaker Public Hub [required]")
@click.option("--technique", help="Customization technique [required]")
@click.option("--instance-type", help="Instance type [required]")
@_hyperpod_telemetry_emitter(Feature.HYPERPOD_CLI, "create_finetuningjob_cli")
@handle_cli_exceptions()
def create_fine_tuning_job_interactive(model_name: str, technique: str, instance_type: str) -> bool:
    """Create a fine-tuning job from recipes with interactive session"""
    if not model_name or not technique or not instance_type:
        click.secho("❌ --model-name, --technique, and --instance-type are required for fine-tuning-job", fg="red")
        return False
    
    try:
        sagemaker_client = _get_sagemaker_client()
        s3_client = _get_s3_client()
        
        # Fetch and validate recipe
        matching_recipe = _fetch_recipe_from_hub(sagemaker_client, model_name, technique, instance_type)
        
        # Get override spec from S3
        override_params_uri = matching_recipe.get('HpEksOverrideParamsS3Uri')
        if not override_params_uri:
            click.secho("❌ Missing override params URI", fg="red")
            return False
        
        spec = _download_s3_json(s3_client, override_params_uri)
        
        # Interactive configuration
        click.secho(f"\n🚀 Interactive Fine-Tuning Job Creation", fg="green", bold=True)
        click.secho(f"Model: {model_name}", fg="blue")
        click.secho(f"Technique: {technique}", fg="blue") 
        click.secho(f"Instance Type: {instance_type}", fg="blue")
        click.secho("\nConfigure the following parameters:\n", fg="yellow")
        
        # Collect all parameters interactively
        config_data = _collect_all_parameters_interactively(spec)
        
        # Get k8s template and render
        k8s_template_uri = matching_recipe.get('HpEksPayloadTemplateS3Uri')
        if not k8s_template_uri:
            click.secho("❌ Missing k8s template URI", fg="red")
            return False
        
        k8s_content = _download_s3_content(s3_client, k8s_template_uri)
        rendered = _render_k8s_template(k8s_content, config_data)
        
        # Submit to Kubernetes
        custom_api = _get_k8s_custom_client()
        click.secho("🚀 Submitting job to Kubernetes...", fg="yellow")
        _submit_k8s_resources(custom_api, rendered)
        
        click.secho("✅ Fine-tuning job created successfully!", fg="green", bold=True)
        return True
        
    except Exception as e:
        try:
            resource_name = config_data.get('name', 'unknown') if 'config_data' in locals() else 'unknown'
            handle_exception(e, resource_name, 'default')
        except Exception as handled_e:
            click.secho(f"❌ {handled_e}", fg="red")
        return False
