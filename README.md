# Log-odds-ratio with Informative Dirichlet priors
This is an implementation based on the paper [Fightin’ Words: Lexical Feature Selection and
Evaluation for Identifying the Content of Political
Conflict](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf).

This is used for the language modeling for stance detection in the paper - [Knowledge Enhanced Masked Language Model for Stance Detection](https://www.aclweb.org/anthology/2021.naacl-main.376/).

Please see [our stance detection repo](https://github.com/GU-DataLab/stance-detection-KE-MLM) 🚀

## Usage
1. Run the following commands.
```shell
python log_odds_ratio.py \
    --filepath_corpus_i=$FP_CORPUS_I \
    --filepath_corpus_j=$FP_CORPUS_J \
    --filepath_background_corpus=$BACKGROUND_CORPUS
```
2. Among generated files, check out the `z_scores.txt` containing words sorted by Z-score. The top words more likely belong to corpus `I` while the botton words likely belong to corpus `J`, with respect to the background corpus.
