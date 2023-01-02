[./main]: ./main.py
[./Makefile]: ./Makefile
[terrier]: https://github.com/terrier-org/terrier-core
[pyterrire]: https://github.com/terrier-org/pyterrier
# IR with PyTerrier

My play-around IR project with [PyTerrier][pyterrier]/[Terrier][terrier].

This IR system accepts SGML formated document files, indexes them and returns
searches for given XML formatted topics. The output is top 1000 documents for
each topic in an TREC qrel format. So for evaluation you can use TREC's
software.

The system is setup for Czech and English document set, though can be easily
adjusted for more languages.

There are numerous experiments testing different setups called *runs*. You can
therefore experiment with different runs.

Detailed description of the different experiments can be found in the
[report][]. TODO:

## Installation

This project is composed of two codebases: java and python.

#### Java installation

Java code is in `java/terrier_ir` and should be easily installable with maven:
```bash
cd java/terrier_ir
mvn package
```

Python calls this code and expects the jar to be at
`java/terrier_ir/target/terrier_ir-1.0.jar`, where it should be after running the
above command.

#### Python installation

Python code is installed using:
```bash
pip install -r requirements.txt
```

## Running experiments

The main program is in [main.py][./main] and has various options:

```bash
python main.py --help
```

There is a [Makefile][./Makefile] to simplify running experiments.

To run a single experiment for both languages run:

```bash
make run-all run=<name_of_the_run>
```

To get all available names of runs run:
```bash
python main.py --list_runs
```

Note that when specifying run name for make ommit the last "\_cs" or "\_en" part.
To run an experiment for only one specific language use the following form.

```bash
make run-all run=<name_of_the_run> lan=<name_of_the_language>
```
