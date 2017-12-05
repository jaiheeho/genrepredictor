#!/usr/bin/env bash
echo "Training unigram"
python train.py unigram
echo "Test with unigram"
python test.py unigram test

echo "Training Bigram"
python train.py bigram
echo "Test with unigram"
python test.py bigram test


echo "Training tfidf"
python train.py tfidf
echo "Test with unigram"
python test.py tfidf test

