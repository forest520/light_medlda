#pragma once

#include "util.h"

DECLARE_int32(num_topic);
const static int MAXINT32 = 0x7fffffff;
static int _average;
static std::vector<int> _line; // [poor; rich]

// Alias table using int32 map (courtesy: jinhui yuan).
class AliasUrn {
public:
  AliasUrn() {
    _average = MAXINT32 / FLAGS_num_topic + 1;
    _line.resize(FLAGS_num_topic);
    urn_.resize(FLAGS_num_topic);
  }

  void Build(const EArray& prob) {
    int ptr = 0, sep = FLAGS_num_topic, urn_sum = 0;
    auto mass = prob.sum();
    for (int i = 0; i < FLAGS_num_topic; ++i) {
      urn_[i].val_ = (int)(prob[i] / mass * MAXINT32);
      urn_sum += urn_[i].val_;
      if (i == FLAGS_num_topic - 1) { urn_[i].val_ += MAXINT32 - urn_sum; }
      if (urn_[i].val_ < _average) { _line[ptr++] = i; }
      else                         { _line[--sep] = i; }
    } // [0,sep) are poor, [sep,end) are rich
    for (int i = 0; i != sep; ++i) {
      auto poor = _line[i], rich = _line.back();
      urn_[poor].ind_ = rich;
      urn_[rich].val_ -= (_average - urn_[poor].val_); // Robin-Hood
      if (urn_[rich].val_ < _average) {
        std::swap(_line.back(), _line[sep]);
        if (sep < FLAGS_num_topic - 1) ++sep;
      }
    }
  }

  int Next() {
    int x = DICE(FLAGS_num_topic);
    int v = DICE(_average);
    return (v <= urn_[x].val_) ? x : urn_[x].ind_;
  }
private:
  struct AliasPair { int ind_, val_; };
  std::vector<AliasPair> urn_;
};
