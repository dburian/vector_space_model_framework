ASSIGNMENT_DIR := ./A1
TREC_EVAL_BIN := ./trec_eval
RES_DIR := ./results
EVAL_DIR := ./evals

.PHONY: run-all run-rm slopes report beamer results

%_train_en.eval: %_train_en.res
	$(TREC_EVAL_BIN) -M1000 $(ASSIGNMENT_DIR)/qrels-train_en.txt $(RES_DIR)/$^ > $(EVAL_DIR)/$@

%_train_cs.eval: %_train_cs.res
	$(TREC_EVAL_BIN) -M1000 $(ASSIGNMENT_DIR)/qrels-train_cs.txt $(RES_DIR)/$^ > $(EVAL_DIR)/$@

%_train_en.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-train_en.xml -d $(ASSIGNMENT_DIR)/documents_en.lst -r $(subst .res,,$(subst _train,,$@)) -o $(RES_DIR)/$@

%_test_en.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-test_en.xml -d $(ASSIGNMENT_DIR)/documents_en.lst -r $(subst .res,,$(subst _test,,$@)) -o $(RES_DIR)/$@

%_train_cs.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-train_cs.xml -d $(ASSIGNMENT_DIR)/documents_cs.lst -r $(subst .res,,$(subst _train,,$@)) -o $(RES_DIR)/$@

%_test_cs.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-test_cs.xml -d $(ASSIGNMENT_DIR)/documents_cs.lst -r $(subst .res,,$(subst _test,,$@)) -o $(RES_DIR)/$@


lan ?= cs en

run-all: $(foreach l,$(lan), $(run)_train_$l.eval $(run)_test_$l.res)

run-rm:
	rm $(EVAL_DIR)/$(run)_train_en.eval $(EVAL_DIR)/$(run)_train_cs.eval
	rm $(RES_DIR)/$(run)_test_en.res $(RES_DIR)/$(run)_train_en.res
	rm $(RES_DIR)/$(run)_test_cs.res $(RES_DIR)/$(run)_train_cs.res

slopes:
	python main.py -q A1/topics-train_$(lan).xml -d A1/documents_$(lan).lst -o supplementary/$(run)_slopes_$(lan).csv --slopes 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --run $(run)_$(lan) --qrels A1/qrels-train_$(lan).txt --pivoted_wmodel cz.dburian.ir.terrier.TfIdfLengthPivoted
	python main.py -q A1/topics-train_$(lan).xml -d A1/documents_$(lan).lst -o supplementary/$(run)_robertson_slopes_$(lan).csv --slopes 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --run $(run)_$(lan) --qrels A1/qrels-train_$(lan).txt --pivoted_wmodel cz.dburian.ir.terrier.TfIdfRobertsonPivoted

all_report_files = $(wildcard report/*.tex)
report: ./report/report.tex  $(all_report_files)
	pdflatex -output-directory report $<
	cd report && bibtex report.aux
	pdflatex -output-directory report $<
	pdflatex -output-directory report $<

all_beamer_files = $(wildcard beamer/*.tex)
beamer: ./beamer/beamer.tex $(all_beamer_files)
	pdflatex -output-directory beamer $<
	pdflatex -output-directory beamer $<

results:
	python scripts/table.py -i $(run) -o supplementary/$(name)_table.tex
	python scripts/graph.py -i $(run) -o supplementary/$(name)_graph.png

submission.zip: beamer report
	zip submission.zip -r src results evals java main.py requirements.txt Makefile README.md scripts pyterrier_home ./report/report.pdf ./beamer/beamer.pdf
	zip submission.zip -d "*/target*" -r "*/__pycache__*"


