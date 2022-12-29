ASSIGNMENT_DIR := ./A1
TREC_EVAL_BIN := ./trec_eval
RES_DIR := ./results
EVAL_DIR := ./evals

%-train_en.eval: %-train_en.res
	$(TREC_EVAL_BIN) -M1000 $(ASSIGNMENT_DIR)/qrels-train_en.txt $(RES_DIR)/$^ > $(EVAL_DIR)/$@

%-train_cs.eval: %-train_cs.res
	$(TREC_EVAL_BIN) -M1000 $(ASSIGNMENT_DIR)/qrels-train_cs.txt $(RES_DIR)/$^ > $(EVAL_DIR)/$@

%-train_en.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-train_en.xml -d $(ASSIGNMENT_DIR)/documents_en.lst -r $(subst .res,,$(subst -train,,$@)) -o $(RES_DIR)/$@

%-test_en.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-test_en.xml -d $(ASSIGNMENT_DIR)/documents_en.lst -r $(subst .res,,$(subst -test,,$@)) -o $(RES_DIR)/$@

%-train_cs.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-train_cs.xml -d $(ASSIGNMENT_DIR)/documents_cs.lst -r $(subst .res,,$(subst -train,,$@)) -o $(RES_DIR)/$@

%-test_cs.res:
	python main.py -q $(ASSIGNMENT_DIR)/topics-test_cs.xml -d $(ASSIGNMENT_DIR)/documents_cs.lst -r $(subst .res,,$(subst -test,,$@)) -o $(RES_DIR)/$@


