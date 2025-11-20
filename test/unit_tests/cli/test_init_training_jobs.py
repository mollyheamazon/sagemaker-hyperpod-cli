"""
Unit tests for init command training job functionality
"""
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import click
from click.testing import CliRunner

from sagemaker.hyperpod.cli.commands.init import init


class TestInitTrainingJobCommands:
    """Test cases for init command with training job templates"""

    def test_init_fine_tuning_job_template_choice(self):
        """Test that fine-tuning-job is available as a template choice"""
        runner = CliRunner()
        result = runner.invoke(init, ['--help'])
        
        assert result.exit_code == 0
        assert 'fine-tuning-job' in result.output

    def test_init_evaluation_job_template_choice(self):
        """Test that evaluation-job is available as a template choice"""
        runner = CliRunner()
        result = runner.invoke(init, ['--help'])
        
        assert result.exit_code == 0
        assert 'evaluation-job' in result.output

    @patch('sagemaker.hyperpod.cli.commands.init._init_training_job')
    def test_init_fine_tuning_job_with_required_params(self, mock_init_training):
        """Test init fine-tuning-job with all required parameters"""
        mock_init_training.return_value = True
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(init, [
                'fine-tuning-job', temp_dir,
                '--model-name', 'test-model',
                '--technique', 'lora',
                '--instance-type', 'ml.p4d.24xlarge'
            ])
            
            assert result.exit_code == 0
            mock_init_training.assert_called_once_with(
                temp_dir, 'fine-tuning-job', 'test-model', 'lora', 'ml.p4d.24xlarge'
            )
            assert "Fine Tuning Job initialized successfully" in result.output


    @patch('sagemaker.hyperpod.cli.commands.init._init_training_job')
    def test_init_evaluation_job_with_required_params(self, mock_init_training):
        """Test init evaluation-job with required parameters"""
        mock_init_training.return_value = True
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(init, [
                'evaluation-job', temp_dir,
                '--model-name', 'test-model',
                '--instance-type', 'ml.p4d.24xlarge'
            ])
            
            assert result.exit_code == 0
            mock_init_training.assert_called_once_with(
                temp_dir, 'evaluation-job', 'test-model', None, 'ml.p4d.24xlarge'
            )
            assert "Evaluation Job initialized successfully" in result.output


    @patch('sagemaker.hyperpod.cli.commands.init._init_training_job')
    def test_init_training_job_without_instance_type(self, mock_init_training):
        """Test init training job without instance type (should trigger interactive selection)"""
        mock_init_training.return_value = True
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(init, [
                'evaluation-job', temp_dir,
                '--model-name', 'test-model'
            ])
            
            assert result.exit_code == 0
            mock_init_training.assert_called_once_with(
                temp_dir, 'evaluation-job', 'test-model', None, None
            )

    def test_init_fine_tuning_job_missing_model_name(self):
        """Test init fine-tuning-job without required model-name parameter"""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(init, [
                'fine-tuning-job', temp_dir,
                '--technique', 'lora',
                '--instance-type', 'ml.p4d.24xlarge'
            ])
            
            assert result.exit_code == 0
            assert "❌ --model-name is required for fine-tuning-job" in result.output

    def test_init_fine_tuning_job_missing_technique(self):
        """Test init fine-tuning-job without required technique parameter"""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(init, [
                'fine-tuning-job', temp_dir,
                '--model-name', 'test-model',
                '--instance-type', 'ml.p4d.24xlarge'
            ])
            
            assert result.exit_code == 0
            assert "❌ --technique is required for fine-tuning-job" in result.output

    def test_init_evaluation_job_missing_model_name(self):
        """Test init evaluation-job without required model-name parameter"""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(init, [
                'evaluation-job', temp_dir,
                '--instance-type', 'ml.p4d.24xlarge'
            ])
            
            assert result.exit_code == 0
            assert "❌ --model-name is required for evaluation-job" in result.output

    @patch('sagemaker.hyperpod.cli.commands.init._init_training_job')
    def test_init_training_job_failure(self, mock_init_training):
        """Test init training job when initialization fails"""
        mock_init_training.return_value = False
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(init, [
                'fine-tuning-job', temp_dir,
                '--model-name', 'test-model',
                '--technique', 'lora',
                '--instance-type', 'ml.p4d.24xlarge'
            ])
            
            assert result.exit_code == 0
            # Should not show success message when initialization fails
            assert "Fine Tuning Job initialized successfully" not in result.output


class TestInitUtilsChanges:
    """Test cases for changes in init_utils.py"""

    def test_is_dynamic_template_fine_tuning_job(self):
        """Test is_dynamic_template recognizes fine-tuning-job"""
        from sagemaker.hyperpod.cli.init_utils import is_dynamic_template
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            override_file = temp_path / ".override_spec.json"
            override_file.write_text('{"test": "data"}')
            
            assert is_dynamic_template("fine-tuning-job", temp_path) is True

    def test_is_dynamic_template_evaluation_job(self):
        """Test is_dynamic_template recognizes evaluation-job"""
        from sagemaker.hyperpod.cli.init_utils import is_dynamic_template
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            override_file = temp_path / ".override_spec.json"
            override_file.write_text('{"test": "data"}')
            
            assert is_dynamic_template("evaluation-job", temp_path) is True

    def test_is_dynamic_template_other_template(self):
        """Test is_dynamic_template returns False for other templates"""
        from sagemaker.hyperpod.cli.init_utils import is_dynamic_template
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            override_file = temp_path / ".override_spec.json"
            override_file.write_text('{"test": "data"}')
            
            assert is_dynamic_template("hyp-pytorch-job", temp_path) is False

    def test_is_dynamic_template_no_override_file(self):
        """Test is_dynamic_template returns False when no override file exists"""
        from sagemaker.hyperpod.cli.init_utils import is_dynamic_template
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            assert is_dynamic_template("fine-tuning-job", temp_path) is False


class TestInitConstantsChanges:
    """Test cases for changes in init_constants.py"""

    def test_templates_include_new_job_types(self):
        """Test that TEMPLATES includes new job types"""
        from sagemaker.hyperpod.cli.constants.init_constants import TEMPLATES
        
        assert "fine-tuning-job" in TEMPLATES
        assert "evaluation-job" in TEMPLATES

    def test_new_job_types_have_dynamic_type(self):
        """Test that new job types are marked as dynamic"""
        from sagemaker.hyperpod.cli.constants.init_constants import TEMPLATES
        
        assert TEMPLATES["fine-tuning-job"]["type"] == "dynamic"
        assert TEMPLATES["evaluation-job"]["type"] == "dynamic"
