import sys
from pathlib import Path
import pytest
from preprocess_data import main
from tests.test_tools.utils import build_args, create_testconfig


class TestPreprocessData:
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    @pytest.fixture
    def output_dir(self, tmp_path, request):
        sys.argv.append('--output-prefix')
        prefix = request.getfixturevalue('prefix')
        sys.argv.append(f'{tmp_path}/{prefix}')
        yield tmp_path

    @pytest.mark.parametrize("params, prefix", test_config["test_preprocess_pretrain_data"])
    def test_preprocess_pretrain_data(self, build_args, output_dir, params, prefix):
        main()
        assert len(list(output_dir.glob(f'{prefix}_text_document.bin'))) == 1
        assert len(list(output_dir.glob(f'{prefix}_text_document.idx'))) == 1
