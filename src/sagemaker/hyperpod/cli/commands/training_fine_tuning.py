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
import shutil

_sagemaker_client = None
_s3_client = None
_k8s_custom_client = None

def get_sagemaker_client():
    global _sagemaker_client
    if _sagemaker_client is None:
        _sagemaker_client = boto3.client(
            "sagemaker",
            endpoint_url="https://sagemaker.beta.us-west-2.ml-platform.aws.a2z.com"
        )
    return _sagemaker_client

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client

def get_k8s_custom_client():
    """Get Kubernetes custom objects API client for PyTorchJob resources."""
    global _k8s_custom_client
    if _k8s_custom_client is None:
        try:
            config.load_kube_config()
        except config.ConfigException:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                raise Exception("Could not configure kubernetes python client")
        _k8s_custom_client = client.CustomObjectsApi()
    return _k8s_custom_client


def _init_fine_tuning_job(directory: str, model_name: str, technique: str, instance_type: str) -> bool:
    """Initialize fine-tuning job configuration."""
    try:
        # Get clients
        sagemaker_client = boto3.client(
            "sagemaker",
            endpoint_url="https://sagemaker.beta.us-west-2.ml-platform.aws.a2z.com"
        )
        s3_client = boto3.client("s3")
        
        # Fetch recipe
        request = {
            "HubName": "SageMakerPublicHub",
            "HubContentType": "Model",
            "HubContentName": model_name
        }
        
        describe_response = sagemaker_client.describe_hub_content(**request)
        hub_content_doc = json.loads(describe_response.get('HubContentDocument', '{}'))
        recipe_collection = hub_content_doc.get('RecipeCollection', [])
        
        matching_recipe = None
        for recipe in recipe_collection:
            if recipe.get('CustomizationTechnique') == technique:
                matching_recipe = recipe
                break
        
        if not matching_recipe:
            click.secho(f"❌ No recipe found for technique: {technique}", fg="red")
            return False
        
        if instance_type not in matching_recipe.get('SupportedInstanceTypes', []):
            click.secho(f"❌ Instance type {instance_type} not supported for this model and technique", fg="red")
            click.secho(f"Supported instance types: {matching_recipe.get('SupportedInstanceTypes', [])}", fg="red")
            return False

        override_params_uri = matching_recipe.get('HpEksOverrideParamsS3Uri')
        k8s_template_uri = matching_recipe.get('HpEksPayloadTemplateS3Uri')
        
        if not override_params_uri or not k8s_template_uri:
            click.secho("❌ Missing S3 URIs in recipe", fg="red")
            return False
        
        # Create directory
        dir_path = Path(directory).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Download override params
        override_bucket = override_params_uri.split('/')[2]
        override_key = '/'.join(override_params_uri.split('/')[3:])
        override_obj = s3_client.get_object(Bucket=override_bucket, Key=override_key)
        override_data = json.loads(override_obj['Body'].read())
        
        # Save override spec
        with open(dir_path / '.override_spec.json', 'w') as f:
            json.dump(override_data, f, indent=2)
        
        # Create config.yaml
        _generate_dynamic_config_yaml(dir_path, "fine-tuning-job", model_name=model_name, technique=technique, instance_type=instance_type)
        
        # Download k8s template
        k8s_bucket = k8s_template_uri.split('/')[2]
        k8s_key = '/'.join(k8s_template_uri.split('/')[3:])
        k8s_obj = s3_client.get_object(Bucket=k8s_bucket, Key=k8s_key)
        k8s_content = k8s_obj['Body'].read().decode('utf-8')
        
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
        click.secho(f"❌ Value required for option: {option}", fg="red")
        ctx.exit(1)
    
    # Validate and convert value
    option_spec = spec[option]
    value_type = option_spec.get("type", "string")
    
    try:
        if value_type == "integer":
            converted_value = int(value)
        elif value_type == "float":
            converted_value = float(value)
        elif value_type == "string":
            converted_value = str(value)
        else:
            converted_value = value
        
        # Validate constraints
        if "min" in option_spec and converted_value < option_spec["min"]:
            click.secho(f"❌ Value {converted_value} is below minimum {option_spec['min']}", fg="red")
            ctx.exit(1)
        
        if "max" in option_spec and converted_value > option_spec["max"]:
            click.secho(f"❌ Value {converted_value} exceeds maximum {option_spec['max']}", fg="red")
            ctx.exit(1)
        
        if "enum" in option_spec and converted_value not in option_spec["enum"]:
            click.secho(f"❌ Value {converted_value} not in allowed values: {option_spec['enum']}", fg="red")
            ctx.exit(1)
        
    except ValueError as e:
        click.secho(f"❌ Invalid value type. Expected {value_type}: {e}", fg="red")
        ctx.exit(1)
    
    # Load and update config.yaml
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{option}:"):
            # Preserve comments and formatting
            indent = len(line) - len(line.lstrip())
            if value_type == "string":
                new_lines.append(f"{' ' * indent}{option}: \"{converted_value}\"\n")
            else:
                new_lines.append(f"{' ' * indent}{option}: {converted_value}\n")
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        click.secho(f"❌ Option {option} not found in config.yaml", fg="red")
        ctx.exit(1)
    
    # Write back to config.yaml
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    click.secho(f"✅ Successfully set {option} = {converted_value}", fg="green")


def _validate_dynamic_template(dir_path: Path):
    """Validate dynamic template config against .override_spec.json"""
    spec_path = dir_path / ".override_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(".override_spec.json not found")
    
    spec = load_dynamic_schema(dir_path)
    config_data = yaml.safe_load((dir_path / "config.yaml").read_text()) or {}
    
    validation_errors = []
    for key, field_spec in spec.items():
        value = config_data.get(key)
        required = field_spec.get("required", False)
        field_type = field_spec.get("type", "string")
        
        if required and (value is None or value == ""):
            validation_errors.append(f"{key}: Required field is missing or empty")
            continue
        
        if value is None:
            continue
        
        # Type validation
        if field_type == "integer" and not isinstance(value, int):
            validation_errors.append(f"{key}: Expected integer, got {type(value).__name__}")
        elif field_type == "float" and not isinstance(value, (int, float)):
            validation_errors.append(f"{key}: Expected number, got {type(value).__name__}")
        elif field_type == "string" and not isinstance(value, str):
            validation_errors.append(f"{key}: Expected string, got {type(value).__name__}")
        
        # Constraint validation
        if isinstance(value, (int, float)):
            if "min" in field_spec and value < field_spec["min"]:
                validation_errors.append(f"{key}: Value {value} below minimum {field_spec['min']}")
            if "max" in field_spec and value > field_spec["max"]:
                validation_errors.append(f"{key}: Value {value} exceeds maximum {field_spec['max']}")
        
        if "enum" in field_spec and value not in field_spec["enum"]:
            validation_errors.append(f"{key}: Value {value} not in allowed values: {field_spec['enum']}")
    
    if validation_errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  • {error}" for error in validation_errors))
    
    return True


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
        template = Template(template_content)
        rendered = template.render(**config_data)
        
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
        
        # Parse and submit to Kubernetes using Python client
        k8s_documents = list(yaml.safe_load_all(rendered))
        custom_api = get_k8s_custom_client()
        
        for k8s_config in k8s_documents:
            if not k8s_config:  # Skip empty documents
                continue
                
            # Extract resource details from the k8s config
            api_version = k8s_config.get('apiVersion', '')
            kind = k8s_config.get('kind', '')
            metadata = k8s_config.get('metadata', {})
            namespace = metadata.get('namespace', 'default')
            
            # Handle standard Kubernetes resources vs custom resources
            if api_version == 'v1' or api_version.startswith('apps/') or api_version.startswith('extensions/'):
                # Standard Kubernetes resource - use CoreV1Api or AppsV1Api
                core_api = client.CoreV1Api()
                
                if kind == 'ConfigMap':
                    core_api.create_namespaced_config_map(namespace=namespace, body=k8s_config)
                elif kind == 'Secret':
                    core_api.create_namespaced_secret(namespace=namespace, body=k8s_config)
                elif kind == 'Service':
                    core_api.create_namespaced_service(namespace=namespace, body=k8s_config)
                else:
                    # For other standard resources, fall back to custom API
                    custom_api.create_namespaced_custom_object(
                        group='',
                        version=api_version,
                        namespace=namespace,
                        plural=kind.lower() + 's',
                        body=k8s_config,
                    )
            else:
                # Custom resource - use CustomObjectsApi
                if '/' in api_version:
                    group, version = api_version.split('/', 1)
                else:
                    group = ''
                    version = api_version
                
                # Convert kind to plural (simple heuristic)
                plural = kind.lower() + 's' if not kind.lower().endswith('s') else kind.lower()
                
                custom_api.create_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    body=k8s_config,
                )
        
        click.secho("✔️ Successfully submitted to Kubernetes", fg="green")
                
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"❌ {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        # Use existing handle_exception for Kubernetes errors
        try:
            # Extract resource name from config for better error messages
            resource_name = config_data.get('name', 'unknown')
            handle_exception(e, resource_name, 'default')
        except Exception as handled_e:
            click.secho(f"❌ {handled_e}", fg="red")
        sys.exit(1)


def _generate_dynamic_config_yaml(dir_path: Path, template: str, version: str = None, model_name: str = None, technique: str = None, instance_type: str = None):
    """Generate config.yaml for dynamic templates with default values"""
    spec = load_dynamic_schema(dir_path)
    
    # Try to preserve existing metadata from current config
    existing_model = model_name
    existing_technique = technique  
    existing_instance_type = instance_type
    
    config_path = dir_path / 'config.yaml'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    if line.startswith('# model: ') and not existing_model:
                        existing_model = line.replace('# model: ', '').strip()
                    elif line.startswith('# fine tune technique: ') and not existing_technique:
                        existing_technique = line.replace('# fine tune technique: ', '').strip()
                    elif line.startswith('# instance type: ') and not existing_instance_type:
                        existing_instance_type = line.replace('# instance type: ', '').strip()
        except:
            pass  # If reading fails, use provided values
    
    with open(config_path, 'w') as f:
        f.write(f"# template: {template}\n")
        if existing_model:
            f.write(f"# model: {existing_model}\n")
        if existing_technique:
            f.write(f"# fine tune technique: {existing_technique}\n")
        if existing_instance_type:
            f.write(f"# instance type: {existing_instance_type}\n")
        f.write("\n")
        
        for key, param_spec in spec.items():
            default_value = param_spec.get('default')
            param_type = param_spec.get('type', 'string')
            min_val = param_spec.get('min')
            max_val = param_spec.get('max')
            description = param_spec.get('description', '')
            required = param_spec.get('required', False)
            
            if description:
                f.write(f"# {description}\n")
            f.write(f"# Type: {param_type}")
            if min_val is not None:
                f.write(f", Min: {min_val}")
            if max_val is not None:
                f.write(f", Max: {max_val}")
            f.write(f", Required: {required}\n")
            
            if default_value is None:
                f.write(f"{key}: null\n\n")
            elif isinstance(default_value, str):
                f.write(f"{key}: {default_value}\n\n")
            elif isinstance(default_value, (list, dict)):
                f.write(f"{key}: {json.dumps(default_value)}\n\n")
            else:
                f.write(f"{key}: {default_value}\n\n")


def _update_config_field(config_path, spec, option, value):
    """Update a single field in config.yaml for dynamic templates"""
    # Validate option exists
    if option not in spec:
        click.secho(f"❌ Unknown option: {option}", fg="red")
        sys.exit(1)
    
    if is_undefined_value(value):
        click.secho(f"❌ Value required for option: {option}", fg="red")
        sys.exit(1)
    
    # Validate and convert value
    option_spec = spec[option]
    value_type = option_spec.get("type", "string")
    
    try:
        if value_type == "integer":
            converted_value = int(value)
        elif value_type == "float":
            converted_value = float(value)
        elif value_type == "string":
            converted_value = str(value)
        else:
            converted_value = value
        
        # Validate constraints
        if "min" in option_spec and converted_value < option_spec["min"]:
            click.secho(f"❌ Value {converted_value} is below minimum {option_spec['min']}", fg="red")
            sys.exit(1)
        
        if "max" in option_spec and converted_value > option_spec["max"]:
            click.secho(f"❌ Value {converted_value} exceeds maximum {option_spec['max']}", fg="red")
            sys.exit(1)
        
        if "enum" in option_spec and converted_value not in option_spec["enum"]:
            click.secho(f"❌ Value {converted_value} not in allowed values: {option_spec['enum']}", fg="red")
            sys.exit(1)
        
    except ValueError as e:
        click.secho(f"❌ Invalid value type. Expected {value_type}: {e}", fg="red")
        sys.exit(1)
    
    # Load and update config.yaml - preserve existing values
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{option}:"):
            # Preserve comments and formatting, no quotes for strings
            indent = len(line) - len(line.lstrip())
            new_lines.append(f"{' ' * indent}{option}: {converted_value}\n")
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        click.secho(f"❌ Option {option} not found in config.yaml", fg="red")
        sys.exit(1)
    
    # Write back to config.yaml
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
