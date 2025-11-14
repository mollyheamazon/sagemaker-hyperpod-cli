import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from kubernetes.client.rest import ApiException
from kubernetes import config

from sagemaker.hyperpod.cli.commands.training_fine_tuning import (
    _configure_dynamic_template,
    _create_dynamic_template,
    _init_fine_tuning_job,
    create_fine_tuning_job_interactive
)
from sagemaker.hyperpod.cli.fine_tuning_utils import (
    _validate_dynamic_template,
    _update_config_field,
    _fetch_recipe_from_hub,
    _validate_and_convert_value,
    _collect_parameter_interactively,
    _get_sagemaker_client,
    _get_s3_client,
    _get_k8s_custom_client,
    _download_s3_json,
    _download_s3_content,
    load_dynamic_schema
)


class TestValidateDynamicTemplate:
    """Test cases for _validate_dynamic_template function"""

    def test_validate_dynamic_template_success(self):
        """Test successful validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create .override_spec.json
            spec = {
                "job_name": {"type": "string", "required": True},
                "epochs": {"type": "integer", "min": 1, "max": 100, "required": False}
            }
            (temp_path / ".override_spec.json").write_text(json.dumps(spec))
            
            # Create valid config.yaml
            config = {"job_name": "test-job", "epochs": 50}
            (temp_path / "config.yaml").write_text(yaml.dump(config))
            
            # Should not raise exception
            result = _validate_dynamic_template(temp_path)
            assert result is True

    def test_validate_dynamic_template_missing_spec(self):
        """Test validation with missing .override_spec.json"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with pytest.raises(FileNotFoundError, match=".override_spec.json not found"):
                _validate_dynamic_template(temp_path)

    def test_validate_dynamic_template_missing_required_field(self):
        """Test validation with missing required field"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create .override_spec.json
            spec = {
                "job_name": {"type": "string", "required": True},
                "epochs": {"type": "integer", "required": False}
            }
            (temp_path / ".override_spec.json").write_text(json.dumps(spec))
            
            # Create config.yaml missing required field
            config = {"epochs": 50}
            (temp_path / "config.yaml").write_text(yaml.dump(config))
            
            with pytest.raises(ValueError, match="job_name: Required field is missing or empty"):
                _validate_dynamic_template(temp_path)


class TestCreateDynamicTemplate:
    """Test cases for _create_dynamic_template function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._validate_dynamic_template')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._submit_k8s_resources')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_k8s_custom_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_create_dynamic_template_success(self, mock_secho, mock_custom_client, mock_submit, mock_validate):
        """Test successful template creation and submission"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create k8s.jinja template
            k8s_template = """---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ job_name }}-config
---
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {{ job_name }}"""
            (temp_path / "k8s.jinja").write_text(k8s_template)
            (temp_path / "config.yaml").write_text("job_name: test-job")
            
            config_data = {"job_name": "test-job"}
            
            # Mock validation success
            mock_validate.return_value = True
            
            # Mock Kubernetes client
            mock_custom_instance = MagicMock()
            mock_custom_client.return_value = mock_custom_instance
            
            # Execute
            _create_dynamic_template(temp_path, config_data)
            
            # Verify validation was called
            mock_validate.assert_called_once_with(temp_path)
            
            # Verify submit was called
            mock_submit.assert_called_once()
            
            # Verify success messages
            mock_secho.assert_any_call("✔️ Configuration validated successfully", fg="green")
            mock_secho.assert_any_call("✔️ Successfully submitted to Kubernetes", fg="green")


class TestInitFineTuningJob:
    """Test cases for _init_fine_tuning_job function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_fine_tuning_job_success(self, mock_secho, mock_get_s3_client, mock_get_sagemaker_client):
        """Test successful fine-tuning job initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock SageMaker client
            mock_sagemaker = MagicMock()
            mock_sagemaker.describe_hub_content.return_value = {
                'HubContentDocument': json.dumps({
                    'RecipeCollection': [{
                        'CustomizationTechnique': 'lora',
                        'SupportedInstanceTypes': ['ml.p4d.24xlarge'],
                        'HpEksOverrideParamsS3Uri': 's3://bucket/override.json',
                        'HpEksPayloadTemplateS3Uri': 's3://bucket/template.yaml'
                    }]
                })
            }
            mock_get_sagemaker_client.return_value = mock_sagemaker
            
            # Mock S3 client
            mock_s3 = MagicMock()
            mock_s3.get_object.side_effect = [
                {'Body': MagicMock(read=lambda: json.dumps({"job_name": {"type": "string", "required": True}}).encode())},
                {'Body': MagicMock(read=lambda: b'apiVersion: v1\nkind: Job')}
            ]
            mock_get_s3_client.return_value = mock_s3
            
            result = _init_fine_tuning_job(temp_dir, "test-model", "lora", "ml.p4d.24xlarge")
            
            assert result is True
            assert Path(temp_dir, ".override_spec.json").exists()
            assert Path(temp_dir, "config.yaml").exists()
            assert Path(temp_dir, "k8s.jinja").exists()


class TestUpdateConfigField:
    """Test cases for _update_config_field function"""

    def test_update_config_field_success(self):
        """Test successful config field update"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("job_name: old-name\nepochs: 10\n")
            
            spec = {
                "job_name": {"type": "string", "required": True},
                "epochs": {"type": "integer", "min": 1, "max": 100}
            }
            
            _update_config_field(config_path, spec, "epochs", "50")
            
            updated_content = config_path.read_text()
            assert "epochs: 50" in updated_content
            assert "job_name: old-name" in updated_content

    def test_update_config_field_validation_error(self):
        """Test config field update with validation error"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("job_name: test\nepochs: 10\n")
            
            spec = {
                "epochs": {"type": "integer", "min": 1, "max": 100}
            }
            
            with pytest.raises(SystemExit):
                _update_config_field(config_path, spec, "epochs", "150")  # Exceeds max


class TestFetchRecipeFromHub:
    """Test cases for _fetch_recipe_from_hub function"""

    def test_fetch_recipe_single_match(self):
        """Test fetching recipe with single matching recipe"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'CustomizationTechnique': 'lora',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge', 'ml.g5.xlarge']
                }]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "lora", "ml.p4d.24xlarge")
        
        assert result['CustomizationTechnique'] == 'lora'
        assert 'ml.p4d.24xlarge' in result['SupportedInstanceTypes']

    def test_fetch_recipe_multiple_recipes_find_match(self):
        """Test fetching recipe with multiple recipes, finding correct one"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [
                    {
                        'CustomizationTechnique': 'lora',
                        'SupportedInstanceTypes': ['ml.g5.xlarge']
                    },
                    {
                        'CustomizationTechnique': 'lora', 
                        'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                    }
                ]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "lora", "ml.p4d.24xlarge")
        
        assert result['CustomizationTechnique'] == 'lora'
        assert result['SupportedInstanceTypes'] == ['ml.p4d.24xlarge']

    def test_fetch_recipe_no_technique_match(self):
        """Test fetching recipe with no matching technique"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'CustomizationTechnique': 'qlora',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        with pytest.raises(ValueError, match="No recipe found for technique: lora"):
            _fetch_recipe_from_hub(mock_client, "test-model", "lora", "ml.p4d.24xlarge")

    def test_fetch_recipe_no_instance_type_match(self):
        """Test fetching recipe with no matching instance type"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [
                    {
                        'CustomizationTechnique': 'lora',
                        'SupportedInstanceTypes': ['ml.g5.xlarge']
                    },
                    {
                        'CustomizationTechnique': 'lora',
                        'SupportedInstanceTypes': ['ml.g4dn.xlarge']
                    }
                ]
            })
        }
        
        with pytest.raises(ValueError, match="Instance type ml.p4d.24xlarge not supported. Supported: \\['ml.g4dn.xlarge', 'ml.g5.xlarge'\\]"):
            _fetch_recipe_from_hub(mock_client, "test-model", "lora", "ml.p4d.24xlarge")


class TestValidateAndConvertValue:
    """Test cases for _validate_and_convert_value function"""

    def test_validate_integer_success(self):
        """Test successful integer validation"""
        result = _validate_and_convert_value("42", {"type": "integer", "min": 1, "max": 100})
        assert result == 42

    def test_validate_integer_type_error(self):
        """Test integer validation with invalid type"""
        with pytest.raises(ValueError, match="Invalid integer value: 'not_a_number'. Please enter a valid integer."):
            _validate_and_convert_value("not_a_number", {"type": "integer"})

    def test_validate_float_success(self):
        """Test successful float validation"""
        result = _validate_and_convert_value("3.14", {"type": "float", "min": 0.0, "max": 10.0})
        assert result == 3.14

    def test_validate_float_min_error(self):
        """Test float validation below minimum"""
        with pytest.raises(ValueError, match="Value -1.0 is below the minimum allowed value of 0.0."):
            _validate_and_convert_value("-1.0", {"type": "float", "min": 0.0})

    def test_validate_float_max_error(self):
        """Test float validation above maximum"""
        with pytest.raises(ValueError, match="Value 15.0 exceeds the maximum allowed value of 10.0."):
            _validate_and_convert_value("15.0", {"type": "float", "max": 10.0})

    def test_validate_enum_success(self):
        """Test successful enum validation"""
        result = _validate_and_convert_value("option1", {"type": "string", "enum": ["option1", "option2"]})
        assert result == "option1"

    def test_validate_enum_error(self):
        """Test enum validation with invalid option"""
        with pytest.raises(ValueError, match="Invalid option 'invalid'. Please choose from: option1, option2."):
            _validate_and_convert_value("invalid", {"type": "string", "enum": ["option1", "option2"]})


class TestCreateFineTuningJobInteractive:
    """Test cases for create_fine_tuning_job_interactive function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_create_missing_parameters(self, mock_secho):
        """Test create command with missing required parameters"""
        # Call the function directly, not the Click command
        from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
        # Get the actual function, not the Click command wrapper
        func = create_fine_tuning_job_interactive.callback
        result = func(None, "lora", "ml.p4d.24xlarge")
        
        assert result is False
        mock_secho.assert_called_with("❌ --model-name, --technique, and --instance-type are required for fine-tuning-job", fg="red")

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_create_missing_technique(self, mock_secho):
        """Test create command with missing technique"""
        from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
        func = create_fine_tuning_job_interactive.callback
        result = func("test-model", None, "ml.p4d.24xlarge")
        
        assert result is False
        mock_secho.assert_called_with("❌ --model-name, --technique, and --instance-type are required for fine-tuning-job", fg="red")

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_create_missing_instance_type(self, mock_secho):
        """Test create command with missing instance type"""
        from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
        func = create_fine_tuning_job_interactive.callback
        result = func("test-model", "lora", None)
        
        assert result is False
        mock_secho.assert_called_with("❌ --model-name, --technique, and --instance-type are required for fine-tuning-job", fg="red")

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_k8s_custom_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._collect_all_parameters_interactively')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._submit_k8s_resources')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_create_success(self, mock_secho, mock_submit, mock_collect, mock_k8s_client, mock_s3_client, mock_sagemaker_client):
        """Test successful create command"""
        from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
        # Mock SageMaker client
        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'CustomizationTechnique': 'lora',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge'],
                    'HpEksOverrideParamsS3Uri': 's3://bucket/override.json',
                    'HpEksPayloadTemplateS3Uri': 's3://bucket/template.yaml'
                }]
            })
        }
        mock_sagemaker_client.return_value = mock_sagemaker
        
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {'Body': MagicMock(read=lambda: json.dumps({"job_name": {"type": "string", "required": True}}).encode())},
            {'Body': MagicMock(read=lambda: b'apiVersion: v1\nkind: Job')}
        ]
        mock_s3_client.return_value = mock_s3
        
        # Mock interactive collection
        mock_collect.return_value = {"job_name": "test-job"}
        
        func = create_fine_tuning_job_interactive.callback
        result = func("test-model", "lora", "ml.p4d.24xlarge")
        
        assert result is True
        mock_secho.assert_any_call("✅ Fine-tuning job created successfully!", fg="green", bold=True)


class TestInitFineTuningJobErrorPaths:
    """Test error paths for _init_fine_tuning_job function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._fetch_recipe_from_hub')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_missing_s3_uris(self, mock_secho, mock_fetch_recipe, mock_get_sagemaker_client, mock_get_s3_client):
        """Test init with missing S3 URIs in recipe"""
        mock_fetch_recipe.return_value = {}
        
        result = _init_fine_tuning_job("test-dir", "model", "technique", "instance")
        
        assert result is False
        mock_secho.assert_called_with("❌ Missing S3 URIs in recipe", fg="red")

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._fetch_recipe_from_hub')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_exception_handling(self, mock_secho, mock_fetch_recipe, mock_get_sagemaker_client, mock_get_s3_client):
        """Test init with exception handling"""
        mock_fetch_recipe.side_effect = Exception("Test error")
        
        result = _init_fine_tuning_job("test-dir", "model", "technique", "instance")
        
        assert result is False
        mock_secho.assert_called_with("❌ Error: Test error", fg="red")


class TestCollectParameterInteractively:
    """Test cases for _collect_parameter_interactively function"""

    @patch('builtins.input', return_value='test_value')
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.click.secho')
    def test_collect_required_parameter(self, mock_secho, mock_input):
        """Test collecting a required parameter"""
        param_spec = {
            'type': 'string',
            'description': 'Test parameter',
            'required': True
        }
        
        key, value = _collect_parameter_interactively('test_param', param_spec)
        
        assert key == 'test_param'
        assert value == 'test_value'

    @patch('builtins.input', return_value='')
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.click.secho')
    def test_collect_optional_parameter_empty(self, mock_secho, mock_input):
        """Test collecting optional parameter with empty input"""
        param_spec = {
            'type': 'string',
            'description': 'Test parameter',
            'required': False
        }
        
        key, value = _collect_parameter_interactively('test_param', param_spec)
        
        assert key == 'test_param'
        assert value is None

    @patch('builtins.input', side_effect=['', 'valid_value'])
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.click.secho')
    def test_collect_required_parameter_retry(self, mock_secho, mock_input):
        """Test collecting required parameter with retry on empty input"""
        param_spec = {
            'type': 'string',
            'description': 'Test parameter',
            'required': True
        }
        
        key, value = _collect_parameter_interactively('test_param', param_spec)
        
        assert key == 'test_param'
        assert value == 'valid_value'
        mock_secho.assert_any_call("❌ This field is required. Please provide a value.", fg="red")

    @patch('builtins.input', return_value='')
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.click.secho')
    def test_collect_parameter_with_default(self, mock_secho, mock_input):
        """Test collecting parameter with default value"""
        param_spec = {
            'type': 'string',
            'description': 'Test parameter',
            'required': False,
            'default': 'default_value'
        }
        
        key, value = _collect_parameter_interactively('test_param', param_spec)
        
        assert key == 'test_param'
        assert value == 'default_value'


class TestClientManagement:
    """Test cases for client management functions"""

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.boto3.client')
    def test_get_sagemaker_client(self, mock_boto3_client):
        """Test SageMaker client creation"""
        # Reset global client
        import sagemaker.hyperpod.cli.fine_tuning_utils as utils
        utils._sagemaker_client = None
        
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        client = _get_sagemaker_client()
        
        assert client == mock_client
        mock_boto3_client.assert_called_once_with(
            "sagemaker",
            endpoint_url="https://sagemaker.gamma.us-west-2.ml-platform.aws.a2z.com"
        )

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.boto3.client')
    def test_get_s3_client(self, mock_boto3_client):
        """Test S3 client creation"""
        # Reset global client
        import sagemaker.hyperpod.cli.fine_tuning_utils as utils
        utils._s3_client = None
        
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        client = _get_s3_client()
        
        assert client == mock_client
        mock_boto3_client.assert_called_once_with("s3")

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.client.CustomObjectsApi')
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.config.load_kube_config')
    def test_get_k8s_client_success(self, mock_load_config, mock_custom_api):
        """Test Kubernetes client creation success"""
        # Reset global client
        import sagemaker.hyperpod.cli.fine_tuning_utils as utils
        utils._k8s_custom_client = None
        
        mock_client = MagicMock()
        mock_custom_api.return_value = mock_client
        
        client = _get_k8s_custom_client()
        
        assert client == mock_client
        mock_load_config.assert_called_once()

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.client.CustomObjectsApi')
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.config.load_incluster_config')
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.config.load_kube_config')
    def test_get_k8s_client_fallback(self, mock_load_config, mock_load_incluster, mock_custom_api):
        """Test Kubernetes client creation with fallback"""
        # Reset global client
        import sagemaker.hyperpod.cli.fine_tuning_utils as utils
        utils._k8s_custom_client = None
        
        mock_load_config.side_effect = config.ConfigException("Config error")
        mock_client = MagicMock()
        mock_custom_api.return_value = mock_client
        
        client = _get_k8s_custom_client()
        
        assert client == mock_client
        mock_load_incluster.assert_called_once()

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.config.load_incluster_config')
    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.config.load_kube_config')
    def test_get_k8s_client_failure(self, mock_load_config, mock_load_incluster):
        """Test Kubernetes client creation failure"""
        # Reset global client
        import sagemaker.hyperpod.cli.fine_tuning_utils as utils
        utils._k8s_custom_client = None
        
        mock_load_config.side_effect = config.ConfigException("Config error")
        mock_load_incluster.side_effect = config.ConfigException("Incluster error")
        
        with pytest.raises(Exception, match="Could not configure kubernetes python client"):
            _get_k8s_custom_client()


class TestCreateDynamicTemplateErrorPaths:
    """Test error paths for _create_dynamic_template function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._validate_dynamic_template')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.sys.exit')
    def test_create_validation_error(self, mock_exit, mock_secho, mock_validate):
        """Test create with validation error"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            mock_validate.side_effect = ValueError("Validation failed")
            
            _create_dynamic_template(temp_path, {})
            
            mock_secho.assert_called_with("❌ Validation failed", fg="red")
            mock_exit.assert_called_with(1)

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.sys.exit')
    def test_create_missing_template(self, mock_exit, mock_secho):
        """Test create with missing k8s.jinja template"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            _create_dynamic_template(temp_path, {})
            
            mock_secho.assert_called_with("❌ .override_spec.json not found", fg="red")
            mock_exit.assert_called_with(1)


class TestDownloadFunctions:
    """Test cases for S3 download functions"""

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.json.loads')
    def test_download_s3_json(self, mock_json_loads):
        """Test S3 JSON download"""
        from sagemaker.hyperpod.cli.fine_tuning_utils import _download_s3_json
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: b'{"key": "value"}')
        }
        mock_json_loads.return_value = {"key": "value"}
        
        result = _download_s3_json(mock_s3_client, "s3://bucket/key.json")
        
        assert result == {"key": "value"}
        mock_s3_client.get_object.assert_called_once_with(Bucket="bucket", Key="key.json")

    def test_download_s3_content(self):
        """Test S3 content download"""
        from sagemaker.hyperpod.cli.fine_tuning_utils import _download_s3_content
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: b'file content')
        }
        
        result = _download_s3_content(mock_s3_client, "s3://bucket/file.txt")
        
        assert result == "file content"
        mock_s3_client.get_object.assert_called_once_with(Bucket="bucket", Key="file.txt")
