import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy

@pytest.fixture
def helper():
    # Only the CLI parser is exercised; no model/GPU/image operations are mocked as successful tests.
    names = ['matplotlib','matplotlib.pyplot','torch','torch.nn','torch.optim','torch.nn.functional','torchvision','workspace_utils','PIL']
    dependencies = {name:MagicMock() for name in names}
    spec = importlib.util.spec_from_file_location('helper', Path(__file__).resolve().parents[1] / 'helper.py')
    module = importlib.util.module_from_spec(spec)
    dependencies['helper'] = module
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module

def test_training_arguments_have_numeric_defaults(helper, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train.py','--data_dir','fixture-images'])
    args = helper.get_input_args()
    assert args.data_dir == 'fixture-images'
    assert args.epochs == 5
    assert args.learning_rate == 0.001
    assert args.device == 'cpu'
    assert args.dropout == 0.5

def test_explicit_training_options(helper, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train.py','--data_dir','images','--epochs','2','--learning_rate','0.02','--arch','densenet121'])
    args = helper.get_input_args()
    assert (args.epochs,args.learning_rate,args.arch) == (2,0.02,'densenet121')

@pytest.mark.parametrize('argv', [[],['--data_dir','images','--epochs','not-a-number']])
def test_invalid_training_arguments_fail(helper, monkeypatch, argv):
    monkeypatch.setattr(sys, 'argv', ['train.py'] + argv)
    with pytest.raises(SystemExit) as error:
        helper.get_input_args()
    assert error.value.code == 2
