--------------------------------------------------------------------------------------------------
         XL-WiC: A Multilingual Benchmark for Evaluating Semantic Contextualization
--------------------------------------------------------------------------------------------------

This package contains the XL-WiC datasets for evaluating multilingual contextualized word representations and an evaluation script.

-----
Datasets are organized per type:
- wic_english/ contains English train and validation datasets from the original WiC dataset (Pilehvar and Camacho-Collados, 2019). This dataset was used for
training in the reference paper.
- xlwic_wikt_monolingual/ contains train, validation and test sets extracted from Wiktionary for each language. Moreover, it contains two extra datasets used 
for analysis purposes (subsets of the test set): IV.test.txt and OOV.test.txt. IV is the In-Vocabulary test set (containaining words that have been seen at training 
time only) and OOV, the Out-Of-Vocabulary test set (containining words that have not been seen at training time).
- xlwic_wn_xlingual/ contains validation and test datasets extracted from WordNet for each language.

The files follow a tab-separated format:
target_word <tab> PoS <tab> start-char-index_1 <tab> end-char-index_1 <tab> start-char-index_2 <tab> end-char-index_2 <tab> example_1 <tab> example_2 <tab> label

- "target_word": the target word which is present in both examples.
- "PoS": the Part-of-Speech tag of the target word (either "N": noun or "V": verb).
- "start-index_i": indicates the start char index of target_word in "i"th example. 
- "end-index_i": indicates the end char index of target_word in "i"th example. 
- "example_i": corresponds to the "i"th example.
- "label": can be 1 or 0 depending on whether the intended sense of the target word is the same in both examples (1) or not (0).

*Note: Test sets are split in "data" and "gold". The "data" files includes all the test instances except for the labels. The "gold" files include the labels 
corresponding to the instance of the same line in the data files.

------

The evaluation scorer (xlwic_scorer.py) can be used as follows from the terminal:

> python xlwic_scorer.py output gold

The output file should contain the labels in the same format as the gold files, with one answer per line (1 if True or 0 if False) corresponding to the data files.

Example usage: 

> python xlwic_scorer.py xlwic_wn/japanese_ja/ja_test_output.txt xlwic_wn/japanese_ja/ja_test_gold.txt

-----

For further details, please see https://pilehvar.github.io/xlwic/ or the reference paper.


====================================================================================================
REFERENCE PAPER
====================================================================================================

When using this dataset, please refer to the following paper:

	Alessandro Raganato, Tommaso Pasini, Jose Camacho-Collados and Mohammad Taher Pilehvar,
	XL-WiC: A Multilingual Benchmark for Evaluating Semantic Contextualization,
	In Proceedings of EMNLP 2020.
	https://www.aclweb.org/anthology/2020.emnlp-main.584/
