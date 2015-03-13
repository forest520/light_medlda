#pragma once

#include "dict.h"
#include "util.h"
#include "alias_urn.h"

#include <string>
#include <vector>

struct Sample {
  struct Pair {
    int tok_, asg_; // token and assignment
    Pair() = default;
    Pair(int tok) : tok_(tok) {}
  };
  IArray label_; // L x 1, {+1,-1}
  EArray aux_; // L x 1, c^2 lambda^-1 in the paper
  std::vector<Pair> body_;
  IArray doc_topic_; // K x 1
};

class Trainer {
public:
  void ReadData(std::string train_file, std::string test_file);
  void Train(); // parameter estimation on training dataset
  void Infer(); // inference on test dataset

private:
  void init_param();
  void build_alias_table();
  void train_one_sample(Sample& doc);
  void draw_classifier();
  void predict();

private:
  std::vector<Sample> train_, test_; // train/test documents
  IMAtrix stat_; // K x V, topic-word count
  IArray  summary_; // K x 1, topic count
  EMatrix classifier_; // L x K, each row is a classifier
  std::vector<AliasUrn> alias_; // alias table for each word
  EMatrix phi_; // K x V
  Dict  dict_;
};
