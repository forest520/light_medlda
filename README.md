Lightweight Gibbs-MedLDA
====

This is a minimalistic C++11 implementation of Light-MedLDA.
It supports multi-label input and single-label prediction.

This package depends on `glog`, `gflags`, `Eigen`, and `gperftools`(optional). To install dependencies,

    cd some_directory
    git clone https://github.com/xunzheng/third_party
    cd third_party
    ./install.sh

Third party libraries will be installed at `some_directory/third_party/`.

Now we can build the main package:

    git clone https://github.com/xunzheng/light_medlda
    cd light_medlda
    ln -s some_directory/third_party .
    make

Toy dataset `20news.train` and `20news.test` is included in the `exp/`
directory. Try

    cd exp
    ./run.sh

to get a sample run.

To see all the available flags, run

    ./med

without any flags.
