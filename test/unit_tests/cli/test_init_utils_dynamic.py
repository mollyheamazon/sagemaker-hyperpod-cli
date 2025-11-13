import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from sagemaker.hyperpod.cli.init_utils import (
    is_dynamic_template,
    load_dynamic_schema
)


class TestIsDynamicTemplate:
    """Test cases for is_dynamic_template function"""

    def test_is_dynamic_template_true(self):
        """Test detection of dynamic template"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create .override_spec.json file
            (temp_path / ".override_spec.json").write_text('{"job_name": {"type": "string"}}')
            
            result = is_dynamic_template("fine-tuning-job", temp_path)
            assert result is True

    def test_is_dynamic_template_false_no_spec_file(self):
        """Test non-dynamic template without spec file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            result = is_dynamic_template("hyp-pytorch-job", temp_path)
            assert result is False


class TestLoadDynamicSchema:
    """Test cases for load_dynamic_schema function"""

    def test_load_dynamic_schema_success(self):
        """Test successful schema loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create .override_spec.json
            spec = {
                "job_name": {"type": "string", "required": True},
                "epochs": {"type": "integer", "min": 1, "max": 100}
            }
            (temp_path / ".override_spec.json").write_text(json.dumps(spec))
            
            result = load_dynamic_schema(temp_path)
            
            assert result == spec
            assert "job_name" in result
            assert result["job_name"]["type"] == "string"

    def test_load_dynamic_schema_file_not_found(self):
        """Test schema loading with missing file returns empty dict"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            result = load_dynamic_schema(temp_path)
            assert result == {}

    def test_load_dynamic_schema_default_path(self):
        """Test with default path (current directory)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory and create spec file there
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                spec = {"test": {"type": "string"}}
                Path(".override_spec.json").write_text(json.dumps(spec))
                
                result = load_dynamic_schema()
                assert result == spec
            finally:
                os.chdir(original_cwd)
