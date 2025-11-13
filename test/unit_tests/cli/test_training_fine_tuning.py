import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from kubernetes.client.rest import ApiException

from sagemaker.hyperpod.cli.commands.training_fine_tuning import (
    _validate_dynamic_template,
    _create_dynamic_template,
    _init_fine_tuning_job,
    _configure_dynamic_template,
    _update_config_field
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
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.get_k8s_custom_client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.client.CoreV1Api')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_create_dynamic_template_success(self, mock_secho, mock_core_api, mock_custom_client, mock_validate):
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
            
            # Mock Kubernetes clients
            mock_core_instance = MagicMock()
            mock_core_api.return_value = mock_core_instance
            mock_custom_instance = MagicMock()
            mock_custom_client.return_value = mock_custom_instance
            
            # Execute
            _create_dynamic_template(temp_path, config_data)
            
            # Verify validation was called
            mock_validate.assert_called_once_with(temp_path)
            
            # Verify success messages
            mock_secho.assert_any_call("✔️ Configuration validated successfully", fg="green")
            mock_secho.assert_any_call("✔️ Successfully submitted to Kubernetes", fg="green")


class TestInitFineTuningJob:
    """Test cases for _init_fine_tuning_job function"""

    @patch('boto3.client')
    @patch('sagemaker.hyperpod.cli.commands.training_fine_tuning.click.secho')
    def test_init_fine_tuning_job_success(self, mock_secho, mock_boto3_client):
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
            
            # Mock S3 client
            mock_s3 = MagicMock()
            mock_s3.get_object.side_effect = [
                {'Body': MagicMock(read=lambda: json.dumps({"job_name": {"type": "string", "required": True}}).encode())},
                {'Body': MagicMock(read=lambda: b'apiVersion: v1\nkind: Job')}
            ]
            
            # Configure boto3.client to return appropriate mocks
            def client_side_effect(service_name, **kwargs):
                if service_name == "sagemaker":
                    return mock_sagemaker
                elif service_name == "s3":
                    return mock_s3
                return MagicMock()
            
            mock_boto3_client.side_effect = client_side_effect
            
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
