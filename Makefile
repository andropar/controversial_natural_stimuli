PYTHON ?= python

.PHONY: smoke compile-core

smoke:
	PYTHONPATH=src $(PYTHON) -c "from pathlib import Path; import cstims; from cstims.paths import model_list_csv, find_share_root; assert find_share_root().exists(); assert model_list_csv().exists(); assert Path('00_stimulus_selection').exists(); print('smoke ok')"

compile-core:
	$(PYTHON) -m compileall -q src 01_brain_model_alignment/code/encoding_model_fitting
