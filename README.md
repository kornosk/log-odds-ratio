# Log-odds-ratio with informative Dirichlet priors
This is an implementation based on the paper [Fightin’ Words: Lexical Feature Selection and
Evaluation for Identifying the Content of Political
Conflict](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)

## Usage
```shell
python log_odds_ratio.py \
    --filepath_corpus_i=$FP_CORPUS_I \
    --filepath_corpus_j=$FP_CORPUS_J \
    --filepath_background_corpus=$BACKGROUND_CORPUS \
    --save_top_words=500
```
