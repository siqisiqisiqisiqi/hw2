python utils_data_processing_1.py $1

python eval.py $2

python bleu_eval.py $1 $2