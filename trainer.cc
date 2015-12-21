#include "trainer.h"

#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

const real NUSQINV         = 1.0;
const int  TRAIN_COLLECT   = 1;
const int  TEST_COLLECT    = 20;
const int  MAX_TEST_BURNIN = 100;
const real GIBBS_CONVERGED = 1e-4;
const real EXP_THRESHOLD   = 640.0;

DEFINE_double(alpha_sum, 6.4, "Parameter of prior on doc-topic distribution");
DEFINE_double(beta, 0.01, "Parameter of prior on topic-word distribution");
DEFINE_double(cost, 1.6, "C param in SVM");
DEFINE_double(ell, 64.0, "Margin param in SVM: usually 1");
DEFINE_int32(num_iter, 50, "Number of burn-in iterations for MCMC");
DEFINE_int32(num_topic, 40, "Model size, usually called K");
DEFINE_int32(eval_every, 100, "Evaluate the model every N iterations");
DEFINE_int32(num_mh, 6, "Number of MH steps for each token");
DEFINE_int32(num_gibbs, 2, "Number of Gibbs sampling for classifier");
DEFINE_int32(top, 10, "Save top N words for each topic");

void Trainer::ReadData(std::string train_file, std::string test_file) {
  size_t num_token = 0;
  FILE *train_fp = fopen(train_file.c_str(), "r"); CHECK_NOTNULL(train_fp);
  char *line = NULL; size_t num_byte;
  while (getline(&line, &num_byte, train_fp) != -1) {
    Sample doc;
    char *ptr = line, *end = line + strlen(line);
    // Read label
    char *sep = strchr(ptr, ':'); // sep at first colon
    while (*sep != ' ') --sep; // sep at space before first colon
    while (ptr != sep) {
      int y = strtol(ptr, &ptr); // ptr at space after label
      int label_id = label_dict_.insert_word(y);
      doc.label_pool_.push_back(label_id);
    }
    // Read text
    while (ptr < end) {
      char *colon = strchr(ptr, ':'); // at colon
      ptr = colon; while (*ptr != ' ') --ptr; // ptr at space before colon
      int word_id = dict_.insert_word(std::string(ptr+1, colon));
      ptr = colon; // ptr at colon
      int count = strtol(++ptr, &ptr); // ptr at space or \n
      for (int i = 0; i < count; ++i)
        doc.body_.emplace_back(word_id);
      num_token += count;
      while (isspace(*ptr)) ++ptr;
    }
    train_.emplace_back(std::move(doc));
  }
  fclose(train_fp);
  LW << "num label: " << label_dict_.size();
  LW << "num train doc: " << train_.size();
  LW << "num train word: " << dict_.size();
  LW << "num train token: " << num_token;
  LI << "---------------------------------------------------------------------";

  // remap labels
  for (auto &doc : train_) {
    doc.label_.setConstant(label_dict_.size(), -1);
    for (auto y : doc.label_pool_) {
      doc.label_(y) = 1;
    }
    doc.label_pool_.clear();
  }

  int num_oov = 0; // oov: out of vocabulary tokens
  size_t num_test_token = 0;
  FILE *test_fp = fopen(test_file.c_str(), "r"); CHECK_NOTNULL(test_fp);
  while (getline(&line, &num_byte, test_fp) != -1) {
    Sample doc;
    char *ptr = line, *end = line + strlen(line);
    // Read label
    doc.label_.setConstant(label_dict_.size(), -1);
    char *sep = strchr(ptr, ':'); // sep at first colon
    while (*sep != ' ') --sep; // sep at space before first colon
    while (ptr != sep) {
      int y = strtol(ptr, &ptr); // ptr at space after label
      int label_id = label_dict_.get_id(y);
      doc.label_(label_id) = 1;
    }
    // Read text, replace oov with random word
    while (ptr < end) {
      char *colon = strchr(ptr, ':'); // at colon
      ptr = colon; while (*ptr != ' ') --ptr; // ptr at space before colon
      int word_id = dict_.get_id(std::string(ptr+1, colon));
      ptr = colon; // ptr at colon
      int count = strtol(++ptr, &ptr); // ptr at space or \n
      for (int i = 0; i < count; ++i) {
        if (word_id < 0) { // replace oov with random word
          doc.body_.emplace_back(DICE(dict_.size()));
          ++num_oov;
        } else {
          doc.body_.emplace_back(word_id);
        }
      }
      num_test_token += count;
      while (isspace(*ptr)) ++ptr;
    }
    test_.emplace_back(std::move(doc));
  }
  fclose(test_fp);
  LW << "num test doc: " << test_.size();
  LW << "num test oov: " << num_oov;
  LW << "num test token: " << num_test_token;
  LI << "---------------------------------------------------------------------";
  free(line);
}

void Trainer::Train() {
  Timer timer;

  // Init
  timer.tic();
  init_param();
  auto init_time = timer.toc();
  LW << "init_param() took " << init_time << " sec, throughput: "
     << (double)summary_.sum() / 1e+6 / init_time << " (M token/sec)";

  // Burn in
  for (int iter = 1; iter <= FLAGS_num_iter; ++iter) {
    timer.tic();
    build_alias_table();
    for (auto& doc : train_) train_one_sample(doc);
    draw_classifier();
    auto iter_time = timer.toc();
    LW << "Iteration " << iter << ", Elapsed time: " << timer.get() << " sec, "
       << "throughput: " << (double)summary_.sum() / 1e+6 / iter_time
       << " (M token/sec)";
    if (iter % FLAGS_eval_every == 0) {
      predict();
      Infer();
    }
  } // end of for each iter

  // Collect samples
  timer.tic();
  EMatrix sum_classifier(label_dict_.size(), FLAGS_num_topic);
  for (int iter = 1; iter <= TRAIN_COLLECT; ++iter) {
    build_alias_table();
    for (auto& doc : train_) train_one_sample(doc);
    draw_classifier();
    sum_classifier += classifier_;
    LW << "Sampling Iteration " << iter;
  } // end of for each iter
  classifier_ = sum_classifier / TRAIN_COLLECT;
  timer.toc();
  LI << "--------------------------------------------------------------";
  LW << "Training Time: " << timer.get() << " sec";

  predict();
  Infer();
}

void Trainer::init_param() {
  stat_.setZero(FLAGS_num_topic, dict_.size());
  summary_.setZero(FLAGS_num_topic);
  classifier_.setZero(label_dict_.size(), FLAGS_num_topic);
  phi_.setZero(FLAGS_num_topic, dict_.size());
  for (auto& doc : train_) {
    doc.aux_.setConstant(label_dict_.size(), 1.0);
    doc.doc_topic_.setZero(FLAGS_num_topic);
    for (auto& pair : doc.body_) {
      pair.asg_ = DICE(FLAGS_num_topic);
      ++doc.doc_topic_(pair.asg_);
      ++stat_(pair.asg_,pair.tok_);
      ++summary_(pair.asg_);
    }
  } // end of for each doc
  alias_.resize(dict_.size());
}

void Trainer::predict() {
  double acc = 0;
  double alpha = FLAGS_alpha_sum / FLAGS_num_topic;
  EArray theta_est(FLAGS_num_topic);
  for (const auto& doc : train_) {
    theta_est = (doc.doc_topic_.cast<real>() + alpha)
                / (doc.body_.size() + FLAGS_alpha_sum);
    int pred = -1;
    (classifier_ * theta_est.matrix()).array().maxCoeff(&pred);
    if (doc.label_(pred) > 0) ++acc;
  } // end of for each doc
  LI << "Train Accuracy (top 1): " << acc / train_.size()
     << " (" << acc << "/" << train_.size() << ")";
}

void Trainer::build_alias_table() {
  Timer alias_timer; alias_timer.tic();
  EArray prob(FLAGS_num_topic);
  EArray denom = summary_.cast<real>() + FLAGS_beta * dict_.size();
  for (int word_id = 0; word_id < dict_.size(); ++word_id) {
    prob = (stat_.col(word_id).cast<real>() + FLAGS_beta) / denom;
    phi_.col(word_id) = prob;
    alias_[word_id].Build(prob);
  }
  //LD << "build_alias_table() took " << alias_timer.toc() << " sec";
}

void Trainer::train_one_sample(Sample& doc) {
  real alpha = FLAGS_alpha_sum / FLAGS_num_topic; // TODO: asymmetric
  real beta = FLAGS_beta;
  real alpha_sum = FLAGS_alpha_sum;
  real beta_sum = beta * dict_.size();
  int  doc_size = doc.body_.size();

  // Draw auxilliary variable
  EArray eta_zd = classifier_ * doc.doc_topic_.matrix().cast<real>();
  EArray zeta = FLAGS_ell - doc.label_.cast<real>() * eta_zd / doc_size;
  EArray mean = FLAGS_cost / zeta.abs();
  real cc = FLAGS_cost * FLAGS_cost;
  doc.aux_ = mean.unaryExpr([cc](real mu){ return draw_invgaussian(mu,cc); });

  // Compute full exponent: A * eta - B * eta^2 - F * eta
  // All L x 1 except the exponent itself which is K x 1
  EArray aux_nn = doc.aux_ / doc_size / doc_size;
  EArray aval = doc.label_.cast<real>()
                * (doc.aux_ * FLAGS_ell + FLAGS_cost) / doc_size;
  EArray bval = aux_nn / 2;
  EArray fval = aux_nn * eta_zd;
  EArray exponent = classifier_.transpose() * (aval - fval).matrix()
                    - classifier_.cwiseAbs2().transpose() * bval.matrix();
  if ((exponent > EXP_THRESHOLD or exponent < -EXP_THRESHOLD).any()) {
    real mx = exponent.maxCoeff();
    real logsum = mx + log((exponent - mx).exp().sum());
    exponent = exponent - logsum;
  }
  AliasUrn exp_alias;
  exp_alias.Build(exponent.exp());

  // Draw topic assignments
  for (int n = 0; n < doc_size; ++n) {
    // Localize
    int word_id   = doc.body_[n].tok_;
    int old_topic = doc.body_[n].asg_;
    eta_zd -= classifier_.col(old_topic).array(); // decrement

    // Mixture of MCMC kernels
    int s = old_topic, t = -1;
    real ntd_alpha, nsd_alpha, ntw_beta, nsw_beta, nt_betasum, ns_betasum;
    real proposal_s, proposal_t, numer, denom, pi, expo, accept;
    EArray sum(label_dict_.size());
    EVector diff(label_dict_.size());
    for (int i = 0; i < FLAGS_num_mh; ++i) {
      auto which_proposal = UNIF01 * 3;
      if (which_proposal < 1) { // word-proposal
        t = alias_[word_id].Next();
        ntd_alpha = doc.doc_topic_(t) + alpha;
        nsd_alpha = doc.doc_topic_(s) + alpha;
        ntw_beta = stat_(t,word_id) + beta;
        nsw_beta = stat_(s,word_id) + beta;
        nt_betasum = summary_(t) + beta_sum;
        ns_betasum = summary_(s) + beta_sum;
        if (t == old_topic) { --ntd_alpha; --ntw_beta; --nt_betasum; }
        if (s == old_topic) { --nsd_alpha; --nsw_beta; --ns_betasum; }
        proposal_t = phi_(t,word_id);
        proposal_s = phi_(s,word_id);
        numer = ntd_alpha * ntw_beta * ns_betasum * proposal_s;
        denom = nsd_alpha * nsw_beta * nt_betasum * proposal_t;
        diff = classifier_.col(t) - classifier_.col(s);
        sum  = classifier_.col(t) + classifier_.col(s);
        expo = (aval - aux_nn * eta_zd - bval * sum).matrix().dot(diff);
      }
      else if (which_proposal < 2) { // doc-proposal
        real ntd_or_alpha = UNIF01 * (doc_size + alpha_sum);
        if (ntd_or_alpha < doc_size) { t = doc.body_[DICE(doc_size)].asg_; }
        else                         { t = DICE(FLAGS_num_topic); }
        ntd_alpha = doc.doc_topic_(t) + alpha;
        nsd_alpha = doc.doc_topic_(s) + alpha;
        ntw_beta = stat_(t,word_id) + beta;
        nsw_beta = stat_(s,word_id) + beta;
        nt_betasum = summary_(t) + beta_sum;
        ns_betasum = summary_(s) + beta_sum;
        if (t == old_topic) { --ntd_alpha; --ntw_beta; --nt_betasum; }
        if (s == old_topic) { --nsd_alpha; --nsw_beta; --ns_betasum; }
        proposal_t = doc.doc_topic_(t) + alpha;
        proposal_s = doc.doc_topic_(s) + alpha;
        numer = ntd_alpha * ntw_beta * ns_betasum * proposal_s;
        denom = nsd_alpha * nsw_beta * nt_betasum * proposal_t;
        diff = classifier_.col(t) - classifier_.col(s);
        sum  = classifier_.col(t) + classifier_.col(s);
        expo = (aval - aux_nn * eta_zd - bval * sum).matrix().dot(diff);
      }
      else { // exp-proposal
        t = exp_alias.Next();
        ntd_alpha = doc.doc_topic_(t) + alpha;
        nsd_alpha = doc.doc_topic_(s) + alpha;
        ntw_beta = stat_(t,word_id) + beta;
        nsw_beta = stat_(s,word_id) + beta;
        nt_betasum = summary_(t) + beta_sum;
        ns_betasum = summary_(s) + beta_sum;
        if (t == old_topic) { --ntd_alpha; --ntw_beta; --nt_betasum; }
        if (s == old_topic) { --nsd_alpha; --nsw_beta; --ns_betasum; }
        numer = ntd_alpha * ntw_beta * ns_betasum;
        denom = nsd_alpha * nsw_beta * nt_betasum;
        diff = classifier_.col(t) - classifier_.col(s);
        expo = (aux_nn * classifier_.col(old_topic).array()).matrix().dot(diff);
      } // end of which proposal
      // Accept/reject
      if (expo > EXP_THRESHOLD or expo < -EXP_THRESHOLD) {
        pi = log(numer) - log(denom) + expo; // log(pi)
        accept = (pi > 0) ? 0.0 : pi; // log(accept)
        s = (log(UNIF01) < accept) ? t : s;
      } else {
        pi = numer / denom * exp(expo);
        accept = (pi > 1.0) ? 1.0 : pi;
        s = (UNIF01 < accept) ? t : s;
      }
    } // end of MH step
    int new_topic = s;
    
    // Update and set
    eta_zd += classifier_.col(new_topic).array(); // increment
    if (new_topic != old_topic) {
      --doc.doc_topic_(old_topic);
      ++doc.doc_topic_(new_topic);
      --stat_(old_topic,word_id);
      ++stat_(new_topic,word_id);
      --summary_(old_topic);
      ++summary_(new_topic);
      doc.body_[n].asg_ = new_topic;
    }
  } // end of for each token
}

/*
void Trainer::draw_classifier() { // exact
  Timer eta_timer; eta_timer.tic();
  EMatrix zmat(FLAGS_num_topic, train_.size()); // K x D
  EVector zd(FLAGS_num_topic);
  for (size_t d = 0; d < train_.size(); ++d) {
    zd.setZero();
    for (auto pair : train_[d].body_) ++zd(pair.asg_);
    zmat.col(d) = zd;
  }
  EVector avec(train_.size()); // diag(D x D)
  EVector svec(train_.size()); // D x 1
  EMatrix precision(FLAGS_num_topic, FLAGS_num_topic);
  EVector vec(FLAGS_num_topic);
  for (int y = 0; y < label_dict_.size(); ++y) {
    for (size_t d = 0; d < train_.size(); ++d) {
      const auto& doc = train_[d];
      real doc_size = doc.body_.size();
      avec(d) = doc.aux_(y) / doc_size / doc_size;
      svec(d) = doc.label_(y)*(doc.aux_(y)*FLAGS_ell+FLAGS_cost)/doc_size;
    } // end of for each doc
    precision = zmat * avec.asDiagonal() * zmat.transpose();
    vec = zmat * svec;
    for (int k = 0; k < FLAGS_num_topic; ++k) precision(k,k) += NUSQINV;
    classifier_.row(y) = draw_mvgaussian(precision, vec);
  } // end of for each y
  LD << "draw_classifier() took " << eta_timer.toc() << " sec";
}
*/

void Trainer::draw_classifier() { // gibbs sampling
  // P = nu I + Q; Q = Z A Z'; P mu = v; v = Z s; U = Z A, w = Z' eta;
  Timer eta_timer; eta_timer.tic();
  EMatrix zmat(train_.size(), FLAGS_num_topic); // D x K
  for (size_t d = 0; d < train_.size(); ++d) {
    zmat.row(d) = train_[d].doc_topic_.cast<real>();
  }
  EVector avec(train_.size()); // diag(D x D)
  EVector svec(train_.size()); // D x 1
  EMatrix umat(train_.size(), FLAGS_num_topic); // D x K
  EVector wvec(train_.size()); // D x 1
  EVector vvec(FLAGS_num_topic); // K x 1
  std::vector<int> k_list(FLAGS_num_topic);
  std::iota(RANGE(k_list), 0);
  for (int y = 0; y < label_dict_.size(); ++y) {
    for (size_t d = 0; d < train_.size(); ++d) {
      const auto& doc = train_[d];
      real doc_size = doc.body_.size();
      avec(d) = doc.aux_(y) / doc_size / doc_size;
      svec(d) = doc.label_(y)*(doc.aux_(y)*FLAGS_ell+FLAGS_cost)/doc_size;
    } // end of for each doc
    umat = avec.asDiagonal() * zmat;
    wvec = zmat * classifier_.row(y).transpose();
    vvec = svec.transpose() * zmat;
    for (int iter = 1; iter <= FLAGS_num_gibbs; ++iter) {
      std::shuffle(RANGE(k_list), _rng);
      for (int k : k_list) {
        real uz = umat.col(k).dot(zmat.col(k));
        real quad = uz + NUSQINV;
        real lin = vvec(k) - umat.col(k).dot(wvec);
        real var = 1.0 / quad;
        real mean = (lin + classifier_(y,k) * uz) * var;
        real new_eta = mean + sqrt(var) * _stdnormal(_rng);
        real diff = new_eta - classifier_(y,k);
        wvec += diff * zmat.col(k);
        classifier_(y,k) = new_eta;
      } // end of each k
    } // end of iter
  } // end of for each y
  eta_timer.toc();
  //LD << "draw_classifier() took " << eta_timer.get() << " sec";
}

void Trainer::Infer() { // TODO: use MH
  real alpha = FLAGS_alpha_sum / FLAGS_num_topic; // TODO: asymmetric
  real beta = FLAGS_beta;
  real beta_sum = beta * dict_.size();

  // Cache est of phi
  EMatrix phi(FLAGS_num_topic, dict_.size()); // K x V
  EArray phi_denom = summary_.cast<real>() + beta_sum;
  for (int word_id = 0; word_id < dict_.size(); ++word_id) {
    phi.col(word_id) = (stat_.col(word_id).cast<real>() + beta) / phi_denom;
  }

  // Multi-label prediction file, HACK
  std::string pred_file = "prediction_k" + std::to_string(FLAGS_num_topic);
  FILE *pred_fp = fopen(pred_file.c_str(), "w"); CHECK_NOTNULL(pred_fp);

  // Go!
  real acc = 0;
  EArray prob(FLAGS_num_topic);
  for (auto& doc : test_) {
    // Initialize to most probable topic assignments
    doc.doc_topic_.setZero(FLAGS_num_topic);
    for (auto& pair : doc.body_) {
      int most_probable_topic = -1;
      stat_.col(pair.tok_).maxCoeff(&most_probable_topic);
      pair.asg_ = most_probable_topic;
      ++doc.doc_topic_(most_probable_topic);
    }

    // Perform Gibbs sampling to obtain an estimate of theta
    real prev_ll = .0;
    real denom = doc.body_.size() + FLAGS_alpha_sum;
    EVector theta(FLAGS_num_topic);
    for (int iter = 1; iter <= MAX_TEST_BURNIN; ++iter) {
      // Gibbs sampling
      for (size_t n = 0; n < doc.body_.size(); ++n) {
        int word_id = doc.body_[n].tok_;
        int old_topic = doc.body_[n].asg_;
        --doc.doc_topic_(old_topic);
        prob = phi.col(word_id).array() * (doc.doc_topic_.cast<real>() + alpha);
        int new_topic = draw_discrete(prob);
        ++doc.doc_topic_(new_topic);
        doc.body_[n].asg_ = new_topic;
      } // end of for each n
      
      // Check convergence using likelihood of test corpus
      theta = (doc.doc_topic_.cast<real>() + alpha) / denom; // theta
      real loglikelihood = .0;
      for (auto pair : doc.body_)
        loglikelihood += log(phi.col(pair.tok_).dot(theta));
      if (fabs((loglikelihood - prev_ll) / prev_ll) < GIBBS_CONVERGED) break;
      prev_ll = loglikelihood;
    } // end of iter

    // Collect samples
    EArray suff_theta(FLAGS_num_topic);
    for (int iter = 1; iter <= TEST_COLLECT; ++iter) {
      // Gibbs sampling
      for (size_t n = 0; n < doc.body_.size(); ++n) {
        int word_id = doc.body_[n].tok_;
        int old_topic = doc.body_[n].asg_;
        --doc.doc_topic_(old_topic);
        prob = phi.col(word_id).array() * (doc.doc_topic_.cast<real>() + alpha);
        int new_topic = draw_discrete(prob);
        ++doc.doc_topic_(new_topic);
        doc.body_[n].asg_ = new_topic;
        ++suff_theta(new_topic);
      } // end of for each n
    } // end of sampling lag

    suff_theta = (suff_theta / TEST_COLLECT + alpha) / denom;
    int pred = -1;
    EArray score = classifier_ * suff_theta.matrix(); // L x 1
    score.maxCoeff(&pred);
    if (doc.label_(pred) > 0) ++acc;

    // Multi-label prediction
    using LabelScorePair = std::pair<int, real>;
    std::vector<LabelScorePair> score_list;
    for (int l = 0; l < label_dict_.size(); ++l) {
      score_list.emplace_back(l, score(l));
    }
    std::sort(RANGE(score_list), [](const LabelScorePair& a, const LabelScorePair& b) {
      return a.second > b.second;
    });
    if ((score > 0).any()) {
      for (const auto& pair : score_list) {
        if (pair.second > 0)
          fprintf(pred_fp, "%d ", pair.first);
      }
      fprintf(pred_fp, "\n");
    } else {
      fprintf(pred_fp, "%d\n", pred); // all negative scores, then put the most confident one
    } // end of if negative

  } // end of for each doc in test set
  LI << "Test Accuracy (top 1): " << acc / test_.size()
     << " (" << acc << "/" << test_.size() << ")";
  fclose(pred_fp);
}

void Trainer::Save(std::string path) {
  double alpha = FLAGS_alpha_sum / FLAGS_num_topic;
  LI << "Saving result in " << path;
  mkdir(path.c_str(), 0777);
  // label mapping
  std::string label_fn = path + "/label";
  std::ofstream label_fs(label_fn);
  CHECK(label_fs.is_open()) << "unable to open " << label_fn;
  for (int y = 0; y < label_dict_.size(); ++y) {
    // new label vs old label
    label_fs << y << "\t" << label_dict_.get_word(y) << std::endl;
  }
  label_fs.close();
  // eta
  std::string eta_fn = path + "/eta";
  std::ofstream eta_fs(eta_fn);
  CHECK(eta_fs.is_open()) << "unable to open " << eta_fn;
  eta_fs << classifier_.transpose() << std::endl;
  eta_fs.close();
  // top words, each column is a list of top words
  EMAtrix topwords(FLAGS_top, FLAGS_num_topic);
  using id_val_t = std::pair<int,real>;
  for (int k = 0; k < FLAGS_num_topic; ++k) {
    std::vector<id_val_t> li;
    for (int word_id = 0; word_id < dict_.size(); ++word_id) {
      li.emplace_back(word_id, phi_(k,word_id));
    }
    std::partial_sort(li.begin(), li.begin() + FLAGS_top, li.end(),
      [](const id_val_t &a, const id_val_t &b){ return a.second > b.second; });
    for (int i = 0; i < FLAGS_top; ++i) {
      topwords(i,k) = li[i].first;
    }
  }
  // salient topics for each label
  for (int y = 0; y < label_dict_.size(); ++y) {
    EArray avg_theta(FLAGS_num_topic);
    int cnt = 0;
    for (const auto &doc : train_) {
      if (doc.label_(y) > 0) {
        ++cnt;
        avg_theta += (doc.doc_topic_.cast<real>() + alpha)
                     / (doc.body_.size() + FLAGS_alpha_sum);
      }
    }
    avg_theta /= cnt;
    std::vector<id_val_t> li;
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      li.emplace_back(k,avg_theta(k));
    }
    std::sort(li.begin(), li.end(),
      [](const id_val_t &a, const id_val_t &b) { return a.second > b.second; });
    std::string salient_fn = path + "/salient." + std::to_string(y);
    FILE *fp = fopen(salient_fn.c_str(), "w");
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      fprintf(fp, "%15s:%02d", "topic", li[k].first);
    }
    fprintf(fp, "\n");
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      fprintf(fp, "     (theta:%3.2f)", li[k].second);
    }
    fprintf(fp, "\n");
    for (int i = 0; i < FLAGS_top; ++i) {
      for (int k = 0; k < FLAGS_num_topic; ++k) {
        fprintf(fp, "%18s", dict_.get_word(topwords(i,li[k].first)).c_str());
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}
