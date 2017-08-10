# tensorwolf
2017 SJTU PPCA project (ML-system)

## How to run

1. make sure you have __numpy__ package in your PYTHONPATH
2. to compile c/c++ kernel, run ```$ make``` in /.../tensorwolf/

## How to test

see [here](https://github.com/merrymercy/dl-system-test) (provided by Mercy...)

## What do I have

basic operators:

| Operator                     | Hint                              |
| ---------------------------- | --------------------------------- |
| +                            | add                               |
| -                            | sub                               |
| *                            | mul                               |
| /                            | div                               |
| matmul                       | multiply two matrix               |
| placeholder                  | To feed later                     |
| reduce_sum                   | get the sum                       |
| reduce_mean                  | get the mean                      |
| global_variables_initializer | initialize all variables          |
| Variable / constant          | literally                         |
| assign                       | A<-B                              |
| exp                          | exp                               |
| log                          | log                               |
| sqrt_op                      | sqrt                              |
| pow_op                       | power                             |
| equal                        | test A==B                         |
| argmax                       | find the index of the maximum     |
| cast                         | cast something's type             |
| pack                         | pack a list of nodes together     |
| reshape                      | change the shape of numpy.ndarray |

some operators for neural network training

| Operator                                 | Hint                              |
| ---------------------------------------- | --------------------------------- |
| zeros                                    | numpy.zeros                       |
| ones                                     | numpy.ones                        |
| float32                                  | numpy.float32                     |
| float64                                  | numpy.float64                     |
| random_normal                            | normal distribution               |
| Session (or 'with' expression)           | something you may call for fun... |
| Session.run                              | to run the computation graph      |
| train.GradientDescentOptimizer().minimize | literally                         |
| train.AdamOptimizer().minimize           | literally                         |
| nn.softmax                               | activation function               |
| nn.relu                                  | activation function               |
| nn.softmax_cross_entropy_with_logits     | a combined operator               |
| nn.conv2d                                | see tensorflow.conv2d             |
| nn.max_pool                              | see tensorflow.max_pool           |
| nn.dropout                               | a method to prevent over-fitting  |

## Versions

- branch __c__:

  nothing special, just compile with gcc and it will be just fine.

- branch __multi-thread__:

  use c++11's __<future>__ std::async to calculate batches in different threads.

## Reference

https://github.com/merrymercy/dl-system-test

http://dlsys.cs.washington.edu/

https://github.com/dlsys-course/assignment2

http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html

https://www.zybuluo.com/hanbingtao/note/485480

https://arxiv.org/abs/1603.04467
