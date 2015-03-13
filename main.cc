#include "trainer.h"

DEFINE_string(train_file, "../dataset/20news.train", "LIBSVM format");
DEFINE_string(test_file, "../dataset/20news.test", "LIBSVM format");
DEFINE_string(dump_prefix, "/tmp/dump", "Prefix for training results");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  if (argc == 1) print_help();
  else print_flags();

  Trainer trainer;
  trainer.ReadData(FLAGS_train_file, FLAGS_test_file);
  trainer.Train();

  return 0;
}

