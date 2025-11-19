import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from kubernetes.client.rest import ApiException
from kubernetes import config
import click

from sagemaker.hyperpod.cli.commands.training_fine_tuning import (
    _configure_dynamic_template,
    _create_dynamic_template,
    _init_training_job,
    # create_fine_tuning_job_interactive
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


class TestInitTrainingJob:
    """Test cases for _init_training_job function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_training_job_success(self, mock_secho, mock_get_s3_client, mock_get_sagemaker_client):
        """Test successful fine-tuning job initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock SageMaker client
            mock_sagemaker = MagicMock()
            mock_sagemaker.describe_hub_content.return_value = {
                'HubContentDocument': json.dumps({
                    'RecipeCollection': [{
                        'Type': 'FineTuning',
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
            
            result = _init_training_job(temp_dir, "fine-tuning-job", "test-model", "lora", "ml.p4d.24xlarge")
            
            assert result is True
            assert Path(temp_dir, ".override_spec.json").exists()
            assert Path(temp_dir, "config.yaml").exists()
            assert Path(temp_dir, "k8s.jinja").exists()

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_pre_training_job_success(self, mock_secho, mock_get_s3_client, mock_get_sagemaker_client):
        """Test successful pre-training job initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock SageMaker client
            mock_sagemaker = MagicMock()
            mock_sagemaker.describe_hub_content.return_value = {
                'HubContentDocument': json.dumps({
                    'RecipeCollection': [{
                        'Type': 'PreTraining',
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
            
            result = _init_training_job(temp_dir, "pre-training-job", "test-model", None, "ml.p4d.24xlarge")
            
            assert result is True
            assert Path(temp_dir, ".override_spec.json").exists()
            assert Path(temp_dir, "config.yaml").exists()
            assert Path(temp_dir, "k8s.jinja").exists()

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_evaluation_job_success(self, mock_secho, mock_get_s3_client, mock_get_sagemaker_client):
        """Test successful evaluation job initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock SageMaker client
            mock_sagemaker = MagicMock()
            mock_sagemaker.describe_hub_content.return_value = {
                'HubContentDocument': json.dumps({
                    'RecipeCollection': [{
                        'Type': 'Evaluation',
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
            
            result = _init_training_job(temp_dir, "evaluation-job", "test-model", None, "ml.p4d.24xlarge")
            
            assert result is True
            assert Path(temp_dir, ".override_spec.json").exists()
            assert Path(temp_dir, "config.yaml").exists()
            assert Path(temp_dir, "k8s.jinja").exists()

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._interactive_cluster_selection')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_training_job_with_interactive_selection(self, mock_secho, mock_get_s3_client, mock_get_sagemaker_client, mock_interactive):
        """Test training job initialization with interactive cluster selection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock interactive selection
            mock_interactive.return_value = ("test-cluster", "ml.p4d.24xlarge")
            
            # Mock SageMaker client
            mock_sagemaker = MagicMock()
            mock_sagemaker.describe_hub_content.return_value = {
                'HubContentDocument': json.dumps({
                    'RecipeCollection': [{
                        'Type': 'FineTuning',
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
            
            # Call without instance_type to trigger interactive selection
            result = _init_training_job(temp_dir, "fine-tuning-job", "test-model", "lora")
            
            assert result is True
            mock_interactive.assert_called_once()
            assert Path(temp_dir, ".override_spec.json").exists()
            assert Path(temp_dir, "config.yaml").exists()
            assert Path(temp_dir, "k8s.jinja").exists()

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._interactive_cluster_selection')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_training_job_interactive_selection_fails(self, mock_secho, mock_get_sagemaker_client, mock_interactive):
        """Test training job initialization when interactive selection fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock interactive selection failure
            mock_interactive.return_value = (None, None)
            
            mock_sagemaker = MagicMock()
            mock_get_sagemaker_client.return_value = mock_sagemaker
            
            result = _init_training_job(temp_dir, "fine-tuning-job", "test-model", "lora")
            
            assert result is False
            mock_interactive.assert_called_once()

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_training_job_framework_support(self, mock_secho, mock_get_s3_client, mock_get_sagemaker_client):
        """Test training job initialization with framework support"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock SageMaker client
            mock_sagemaker = MagicMock()
            mock_sagemaker.describe_hub_content.return_value = {
                'HubContentDocument': json.dumps({
                    'RecipeCollection': [{
                        'Type': 'PreTraining',
                        'Framework': 'CHECKPOINTLESS',
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
            
            result = _init_training_job(temp_dir, "pre-training-job", "test-model", None, "ml.p4d.24xlarge", framework="checkpointless")
            
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

    def test_fetch_recipe_fine_tuning_single_match(self):
        """Test fetching fine-tuning recipe with single matching recipe"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'FineTuning',
                    'CustomizationTechnique': 'lora',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge', 'ml.g5.xlarge']
                }]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "fine-tuning-job", "lora", "ml.p4d.24xlarge")
        
        assert result['Type'] == 'FineTuning'
        assert result['CustomizationTechnique'] == 'lora'
        assert 'ml.p4d.24xlarge' in result['SupportedInstanceTypes']

    def test_fetch_recipe_pre_training_success(self):
        """Test fetching pre-training recipe successfully"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'PreTraining',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "pre-training-job", None, "ml.p4d.24xlarge")
        
        assert result['Type'] == 'PreTraining'
        assert 'ml.p4d.24xlarge' in result['SupportedInstanceTypes']

    def test_fetch_recipe_evaluation_success(self):
        """Test fetching evaluation recipe successfully"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'Evaluation',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "evaluation-job", None, "ml.p4d.24xlarge")
        
        assert result['Type'] == 'Evaluation'
        assert 'ml.p4d.24xlarge' in result['SupportedInstanceTypes']

    def test_fetch_recipe_framework_support(self):
        """Test fetching recipe with framework support"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [
                    {
                        'Type': 'PreTraining',
                        'Framework': 'Standard',
                        'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                    },
                    {
                        'Type': 'PreTraining',
                        'Framework': 'CHECKPOINTLESS',
                        'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                    }
                ]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "pre-training-job", None, "ml.p4d.24xlarge", framework="checkpointless")
        
        assert result['Type'] == 'PreTraining'
        assert result['Framework'] == 'CHECKPOINTLESS'

    def test_fetch_recipe_framework_case_insensitive(self):
        """Test fetching recipe with case insensitive framework"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'PreTraining',
                    'Framework': 'NOVA',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "pre-training-job", None, "ml.p4d.24xlarge", framework="nova")
        
        assert result['Type'] == 'PreTraining'
        assert result['Framework'] == 'NOVA'

    def test_fetch_recipe_no_framework_match(self):
        """Test fetching recipe with no framework match"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'PreTraining',
                    'Framework': 'Standard',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        with pytest.raises(ValueError, match="No CHECKPOINTLESS recipe found for job type: pre-training-job"):
            _fetch_recipe_from_hub(mock_client, "test-model", "pre-training-job", None, "ml.p4d.24xlarge", framework="checkpointless")

    def test_fetch_recipe_without_instance_type(self):
        """Test fetching recipe without specifying instance type"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'PreTraining',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "pre-training-job")
        
        assert result['Type'] == 'PreTraining'
        assert result['SupportedInstanceTypes'] == ['ml.p4d.24xlarge']

    def test_fetch_recipe_multiple_recipes_find_match(self):
        """Test fetching recipe with multiple recipes, finding correct one"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [
                    {
                        'Type': 'FineTuning',
                        'CustomizationTechnique': 'lora',
                        'SupportedInstanceTypes': ['ml.g5.xlarge']
                    },
                    {
                        'Type': 'FineTuning',
                        'CustomizationTechnique': 'lora', 
                        'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                    }
                ]
            })
        }
        
        result = _fetch_recipe_from_hub(mock_client, "test-model", "fine-tuning-job", "lora", "ml.p4d.24xlarge")
        
        assert result['Type'] == 'FineTuning'
        assert result['CustomizationTechnique'] == 'lora'
        assert result['SupportedInstanceTypes'] == ['ml.p4d.24xlarge']

    def test_fetch_recipe_unsupported_job_type(self):
        """Test fetching recipe with unsupported job type"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': []
            })
        }
        
        with pytest.raises(ValueError, match="Unsupported job type: invalid-job"):
            _fetch_recipe_from_hub(mock_client, "test-model", "invalid-job")

    def test_fetch_recipe_no_technique_match(self):
        """Test fetching fine-tuning recipe with no matching technique"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'FineTuning',
                    'CustomizationTechnique': 'qlora',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        with pytest.raises(ValueError, match="No recipe found for technique: lora"):
            _fetch_recipe_from_hub(mock_client, "test-model", "fine-tuning-job", "lora", "ml.p4d.24xlarge")

    def test_fetch_recipe_no_job_type_match(self):
        """Test fetching recipe with no matching job type"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [{
                    'Type': 'FineTuning',
                    'CustomizationTechnique': 'lora',
                    'SupportedInstanceTypes': ['ml.p4d.24xlarge']
                }]
            })
        }
        
        with pytest.raises(ValueError, match="No recipe found for job type: pre-training-job"):
            _fetch_recipe_from_hub(mock_client, "test-model", "pre-training-job", None, "ml.p4d.24xlarge")



    def test_fetch_recipe_no_instance_type_match(self):
        """Test fetching recipe with no matching instance type"""
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            'HubContentDocument': json.dumps({
                'RecipeCollection': [
                    {
                        'Type': 'FineTuning',
                        'CustomizationTechnique': 'lora',
                        'SupportedInstanceTypes': ['ml.g5.xlarge']
                    },
                    {
                        'Type': 'FineTuning',
                        'CustomizationTechnique': 'lora',
                        'SupportedInstanceTypes': ['ml.g4dn.xlarge']
                    }
                ]
            })
        }
        
        with pytest.raises(ValueError, match="Instance type ml.p4d.24xlarge not supported. Supported: \\['ml.g4dn.xlarge', 'ml.g5.xlarge'\\]"):
            _fetch_recipe_from_hub(mock_client, "test-model", "fine-tuning-job", "lora", "ml.p4d.24xlarge")


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


# class TestCreateFineTuningJobInteractive:
#     """Test cases for create_fine_tuning_job_interactive function"""

#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
#     def test_create_missing_parameters(self, mock_secho):
#         """Test create command with missing required parameters"""
#         # Call the function directly, not the Click command
#         from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
#         # Get the actual function, not the Click command wrapper
#         func = create_fine_tuning_job_interactive.callback
#         result = func(None, "lora", "ml.p4d.24xlarge")
        
#         assert result is False
#         mock_secho.assert_called_with("❌ --model-name, --technique, and --instance-type are required for fine-tuning-job", fg="red")

#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
#     def test_create_missing_technique(self, mock_secho):
#         """Test create command with missing technique"""
#         from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
#         func = create_fine_tuning_job_interactive.callback
#         result = func("test-model", None, "ml.p4d.24xlarge")
        
#         assert result is False
#         mock_secho.assert_called_with("❌ --model-name, --technique, and --instance-type are required for fine-tuning-job", fg="red")

#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
#     def test_create_missing_instance_type(self, mock_secho):
#         """Test create command with missing instance type"""
#         from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
#         func = create_fine_tuning_job_interactive.callback
#         result = func("test-model", "lora", None)
        
#         assert result is False
#         mock_secho.assert_called_with("❌ --model-name, --technique, and --instance-type are required for fine-tuning-job", fg="red")

#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_k8s_custom_client')
#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._collect_all_parameters_interactively')
#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._submit_k8s_resources')
#     @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
#     def test_create_success(self, mock_secho, mock_submit, mock_collect, mock_k8s_client, mock_s3_client, mock_sagemaker_client):
#         """Test successful create command"""
#         from sagemaker.hyperpod.cli.commands.training_fine_tuning import create_fine_tuning_job_interactive
        
#         # Mock SageMaker client
#         mock_sagemaker = MagicMock()
#         mock_sagemaker.describe_hub_content.return_value = {
#             'HubContentDocument': json.dumps({
#                 'RecipeCollection': [{
#                     'Type': 'FineTuning',
#                     'CustomizationTechnique': 'lora',
#                     'SupportedInstanceTypes': ['ml.p4d.24xlarge'],
#                     'HpEksOverrideParamsS3Uri': 's3://bucket/override.json',
#                     'HpEksPayloadTemplateS3Uri': 's3://bucket/template.yaml'
#                 }]
#             })
#         }
#         mock_sagemaker_client.return_value = mock_sagemaker
        
#         # Mock S3 client
#         mock_s3 = MagicMock()
#         mock_s3.get_object.side_effect = [
#             {'Body': MagicMock(read=lambda: json.dumps({"job_name": {"type": "string", "required": True}}).encode())},
#             {'Body': MagicMock(read=lambda: b'apiVersion: v1\nkind: Job')}
#         ]
#         mock_s3_client.return_value = mock_s3
        
#         # Mock interactive collection
#         mock_collect.return_value = {"job_name": "test-job"}
        
#         # Mock submit to return True
#         mock_submit.return_value = True
        
#         func = create_fine_tuning_job_interactive.callback
#         result = func("test-model", "lora", "ml.p4d.24xlarge")
        
#         assert result is True
#         mock_secho.assert_any_call("✅ Fine-tuning job created successfully!", fg="green", bold=True)


class TestInitFineTuningJobErrorPaths:
    """Test error paths for _init_training_job function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._fetch_recipe_from_hub')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_missing_s3_uris(self, mock_secho, mock_fetch_recipe, mock_get_sagemaker_client, mock_get_s3_client):
        """Test init with missing S3 URIs in recipe"""
        mock_fetch_recipe.return_value = {}
        
        result = _init_training_job("test-dir", "fine-tuning-job", "model", "technique", "instance")
        
        assert result is False
        mock_secho.assert_called_with("❌ Missing S3 URIs in recipe", fg="red")

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_s3_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._get_sagemaker_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._fetch_recipe_from_hub')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_exception_handling(self, mock_secho, mock_fetch_recipe, mock_get_sagemaker_client, mock_get_s3_client):
        """Test init with exception handling"""
        mock_fetch_recipe.side_effect = Exception("Test error")
        
        result = _init_training_job("test-dir", "fine-tuning-job", "model", "technique", "instance")
        
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
            endpoint_url="https://sagemaker.beta.us-west-2.ml-platform.aws.a2z.com"
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


class TestInstanceTypeOverride:
    """Test cases for instance type override functionality in _generate_dynamic_config_yaml"""

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.load_dynamic_schema')
    @patch('builtins.open', new_callable=mock_open)
    def test_generate_config_with_instance_type_override(self, mock_file, mock_load_schema):
        """Test that instance_type field is overridden with user input"""
        from sagemaker.hyperpod.cli.fine_tuning_utils import _generate_dynamic_config_yaml
        
        # Mock schema with instance_type parameter
        mock_schema = {
            "instance_type": {
                "type": "string",
                "description": "Instance type for training",
                "required": True,
                "default": "ml.g5.2xlarge"
            },
            "model_name": {
                "type": "string", 
                "description": "Model name",
                "required": True
            }
        }
        mock_load_schema.return_value = mock_schema
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Call with instance_type override
            _generate_dynamic_config_yaml(
                temp_path, 
                "fine-tuning-job",
                model_name="test-model",
                technique="SFT", 
                instance_type="ml.g5.48xlarge"
            )
            
            # Verify the file was written with overridden instance_type
            written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
            assert "ml.g5.48xlarge" in written_content
            assert "instance_type: ml.g5.48xlarge" in written_content

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.load_dynamic_schema')
    @patch('builtins.open', new_callable=mock_open)
    def test_generate_config_without_instance_type_override(self, mock_file, mock_load_schema):
        """Test that default instance_type is used when no override provided"""
        from sagemaker.hyperpod.cli.fine_tuning_utils import _generate_dynamic_config_yaml
        
        # Mock schema with instance_type parameter
        mock_schema = {
            "instance_type": {
                "type": "string",
                "description": "Instance type for training", 
                "required": True,
                "default": "ml.g5.2xlarge"
            }
        }
        mock_load_schema.return_value = mock_schema
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Call without instance_type override
            _generate_dynamic_config_yaml(
                temp_path,
                "fine-tuning-job", 
                model_name="test-model",
                technique="SFT"
            )
            
            # Verify the file was written with default instance_type
            written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
            assert "ml.g5.2xlarge" in written_content

    @patch('sagemaker.hyperpod.cli.fine_tuning_utils.load_dynamic_schema')
    @patch('builtins.open', new_callable=mock_open)
    def test_generate_config_instance_type_override_only_affects_instance_type_field(self, mock_file, mock_load_schema):
        """Test that instance_type override only affects the instance_type field, not other fields"""
        from sagemaker.hyperpod.cli.fine_tuning_utils import _generate_dynamic_config_yaml
        
        # Mock schema with multiple parameters
        mock_schema = {
            "instance_type": {
                "type": "string",
                "description": "Instance type for training",
                "required": True,
                "default": "ml.g5.2xlarge"
            },
            "other_param": {
                "type": "string",
                "description": "Other parameter", 
                "required": False,
                "default": "default_value"
            }
        }
        mock_load_schema.return_value = mock_schema
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Call with instance_type override
            _generate_dynamic_config_yaml(
                temp_path,
                "fine-tuning-job",
                model_name="test-model", 
                technique="SFT",
                instance_type="ml.g5.48xlarge"
            )
            
            # Verify instance_type was overridden but other_param uses default
            written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
            assert "ml.g5.48xlarge" in written_content
            assert "default_value" in written_content


class TestInteractiveClusterSelection:
    """Test cases for _interactive_cluster_selection function"""

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._fetch_recipe_from_hub')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_interactive_cluster_selection_no_supported_instance_types(self, mock_secho, mock_fetch_recipe):
        """Test interactive cluster selection with no supported instance types"""
        from sagemaker.hyperpod.cli.commands.training_fine_tuning import _interactive_cluster_selection
        
        # Mock recipe with no supported instance types
        mock_fetch_recipe.return_value = {
            'SupportedInstanceTypes': []
        }
        
        mock_sagemaker_client = MagicMock()
        result = _interactive_cluster_selection(mock_sagemaker_client, "test-model", "fine-tuning-job", "lora")
        
        assert result == (None, None)
        mock_secho.assert_any_call("❌ No supported instance types found in recipe", fg="red")

    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning._fetch_recipe_from_hub')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_interactive_cluster_selection_exception_handling(self, mock_secho, mock_fetch_recipe):
        """Test interactive cluster selection with exception handling"""
        from sagemaker.hyperpod.cli.commands.training_fine_tuning import _interactive_cluster_selection
        
        # Mock recipe fetch to raise exception
        mock_fetch_recipe.side_effect = Exception("Test error")
        
        mock_sagemaker_client = MagicMock()
        result = _interactive_cluster_selection(mock_sagemaker_client, "test-model", "fine-tuning-job", "lora")
        
        assert result == (None, None)
        mock_secho.assert_any_call("❌ Error during cluster selection: Test error", fg="red")


class TestHypCliDeleteCommand:
    """Test cases for the CLI delete command fix"""

    def test_fine_tuning_delete_command_registration(self):
        """Test that fine-tuning delete command is properly registered"""
        from sagemaker.hyperpod.cli.hyp_cli import delete
        
        # Check that fine-tuning-job delete command exists
        commands = delete.list_commands(None)
        assert "fine-tuning-job" in commands
        
        # Get the command and verify it's the delete command, not describe
        fine_tuning_cmd = delete.get_command(None, "fine-tuning-job")
        assert fine_tuning_cmd is not None
        assert "Delete" in fine_tuning_cmd.help or "delete" in fine_tuning_cmd.help.lower()

    def test_fine_tuning_delete_command_help_text(self):
        """Test that fine-tuning delete command has correct help text"""
        from sagemaker.hyperpod.cli.hyp_cli import delete
        
        fine_tuning_cmd = delete.get_command(None, "fine-tuning-job")
        assert fine_tuning_cmd is not None
        assert "Delete a HyperPod fine-tuning job" in fine_tuning_cmd.help

    def test_pre_training_commands_registration(self):
        """Test that pre-training commands are properly registered"""
        from sagemaker.hyperpod.cli.hyp_cli import list, describe, delete, list_pods, get_logs, get_operator_logs
        
        # Check list command
        commands = list.list_commands(None)
        assert "pre-training-job" in commands
        
        # Check describe command
        commands = describe.list_commands(None)
        assert "pre-training-job" in commands
        
        # Check delete command
        commands = delete.list_commands(None)
        assert "pre-training-job" in commands
        
        # Check list-pods command
        commands = list_pods.list_commands(None)
        assert "pre-training-job" in commands
        
        # Check get-logs command
        commands = get_logs.list_commands(None)
        assert "pre-training-job" in commands
        
        # Check get-operator-logs command
        commands = get_operator_logs.list_commands(None)
        assert "pre-training-job" in commands

    def test_evaluation_commands_registration(self):
        """Test that evaluation commands are properly registered"""
        from sagemaker.hyperpod.cli.hyp_cli import list, describe, delete, list_pods, get_logs, get_operator_logs
        
        # Check list command
        commands = list.list_commands(None)
        assert "evaluation-job" in commands
        
        # Check describe command
        commands = describe.list_commands(None)
        assert "evaluation-job" in commands
        
        # Check delete command
        commands = delete.list_commands(None)
        assert "evaluation-job" in commands
        
        # Check list-pods command
        commands = list_pods.list_commands(None)
        assert "evaluation-job" in commands
        
        # Check get-logs command
        commands = get_logs.list_commands(None)
        assert "evaluation-job" in commands
        
        # Check get-operator-logs command
        commands = get_operator_logs.list_commands(None)
        assert "evaluation-job" in commands

    def test_pre_training_command_help_texts(self):
        """Test that pre-training commands have correct help text"""
        from sagemaker.hyperpod.cli.hyp_cli import list, describe, delete
        
        # Check list command help
        pre_training_list_cmd = list.get_command(None, "pre-training-job")
        assert pre_training_list_cmd is not None
        assert "List all HyperPod pre-training jobs" in pre_training_list_cmd.help
        
        # Check describe command help
        pre_training_describe_cmd = describe.get_command(None, "pre-training-job")
        assert pre_training_describe_cmd is not None
        assert "Describe a HyperPod pre-training job" in pre_training_describe_cmd.help
        
        # Check delete command help
        pre_training_delete_cmd = delete.get_command(None, "pre-training-job")
        assert pre_training_delete_cmd is not None
        assert "Delete a HyperPod pre-training job" in pre_training_delete_cmd.help

    def test_evaluation_command_help_texts(self):
        """Test that evaluation commands have correct help text"""
        from sagemaker.hyperpod.cli.hyp_cli import list, describe, delete
        
        # Check list command help
        evaluation_list_cmd = list.get_command(None, "evaluation-job")
        assert evaluation_list_cmd is not None
        assert "List all HyperPod evaluation jobs" in evaluation_list_cmd.help
        
        # Check describe command help
        evaluation_describe_cmd = describe.get_command(None, "evaluation-job")
        assert evaluation_describe_cmd is not None
        assert "Describe a HyperPod evaluation job" in evaluation_describe_cmd.help
        
        # Check delete command help
        evaluation_delete_cmd = delete.get_command(None, "evaluation-job")
        assert evaluation_delete_cmd is not None
        assert "Delete a HyperPod evaluation job" in evaluation_delete_cmd.help
