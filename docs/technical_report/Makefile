REPORT_PDF := fp4_training_systems_report.pdf
SECTIONS := $(wildcard sections/*.tex) $(wildcard appendices/*.tex)
SOURCE_DATE_EPOCH ?= 1788307200
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE = 1

.PHONY: all figures arxiv-source clean

all: figures $(REPORT_PDF)

figures:
	python figures/make_report_figures.py
	python figures/build_fusion_schedule_figure.py
	python figures/build_llama8b_160b_figure.py \
		--snapshot-in data/llama8b_160b_loss_snapshot_20260827.csv \
		--seed-adjustment data/llama8b_seed42_loss_adjustment_20260828.csv \
		--pure-v5-recovered data/llama8b_pure_v5_recovered_log_points_20260830.csv \
		--te-f0l4-history data/llama8b_terminal_training_history_r1_results_20260903/llama8b_te_f0l4_seed42_history_20260903.csv \
		--operand-h32-history data/llama8b_terminal_training_history_r1_results_20260903/llama8b_operand_h32_history_20260903.csv \
		--mxfp4-rht-history data/llama8b_mxfp4_rht_history_20260831.csv \
		--localcta-rht-history data/llama8b_localcta_rht_canonical_metrics_20260830.csv \
		--operand-h16-history data/llama8b_operand_h16_history_20260902.csv \
		--figure-out figures/llama8b_160b_training_loss.png \
		--zoom-figure-out figures/llama8b_160b_training_loss_zoom.png
	python figures/build_llama8b_downstream_figure.py \
		--ledger data/llama8b_final_eval_integration_r1_results_20260903/downstream/DOWNSTREAM_EVAL_LEDGER.csv \
		--output figures/fp4_downstream_delta_bf16.png
	python figures/build_llama8b_downstream_table.py \
		--ledger data/llama8b_final_eval_integration_r1_results_20260903/downstream/DOWNSTREAM_EVAL_LEDGER.csv \
		--output data/llama8b_scaledrope_downstream_table_rows_20260903.tex
	python figures/build_llama8b_validation_figure.py \
		--ledger data/llama8b_final_eval_integration_r1_results_20260903/validation/VALIDATION_LEDGER.csv \
		--output figures/llama8b_validation_nll.png

$(REPORT_PDF): figures main.tex references.bib public_report.sty $(SECTIONS)
	latexmk -g -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
	cp main.pdf $(REPORT_PDF)

arxiv-source: $(REPORT_PDF)
	python prepare_arxiv_submission.py

clean:
	latexmk -C main.tex
	rm -rf build/arxiv-source build/fp4_training_systems_arxiv.tar.gz \
		build/fp4_training_systems_arxiv.tar.gz.sha256 \
		build/fp4_training_systems_overleaf.zip \
		build/fp4_training_systems_overleaf.zip.sha256
