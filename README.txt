In order to run the model, download the 'family_ontology.owl' file from the 'data' folder and change the directory path in "create_canonical_model.py".
Then go to 'main.ipynb', select the desired hyperparameters, and run the training/eval routine.

Training and evaluation occur simultaneously, with the evaluation procedure being controlled by the EVAL_FREQ variable, and the EVAL_TRAIN (if turned to true, it evaluates the model on training data).

The options to train without negative sampling and restrict the language are currently deprecated, and will not work if turned to FALSE and TRUE, respectively.

Due to issues with the reasoning process on the ontology, certain parts of the canonical interpretation of the knowledge base had to be hardcoded, which prevents the model from being able to be used with other ontologies out of the box.