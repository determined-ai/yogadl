# Yoga Data Layer: The _Flexible_ Data Layer

A better approach to data loading for Deep Learning.  API-transparent caching to disk, GCS, or S3.

## Why `yogadl`?

At Determined AI, we help many customers perform high-performance data loading for deep learning
models.  We believe every data loader should have two layers: the **random-access layer** and the
**sequential layer**.

The **random-access layer** is critical for good training infrastructure.  Direct random access to
any record enables:

  - Shuffling (potentially every epoch)
  - Pausing/continuing training mid-epoch
  - Sharding the dataset efficiently for distributed training

The **sequential layer** starts as soon as you decide the order in which you will access the records in
the dataset.  Often the transition is implicit, in which case it starts as soon as you are done
modifying the order of access (i.e. via shuffling, sharding, or splitting).  This layer is vital to
performance optimizations because it enables:

  - Prefetching data loading to hide latency costs
  - Parallelizing data loading to hide compute costs

Here is a simple code snippet to illustrate what the transition from random-access layer to
sequential layer looks like:

```python
# Start of random-access layer.
indices = list(range(100))
indices = indices[skip:]
indices=np.random.shuffle(indices)

# Start of sequential layer.

def record_gen():
    for i in indices:
        yield read_file_at_index(i)

record_ds = tf.data.Dataset.from_generator(record_gen, ...)
final_ds = record_ds.prefetch(...)

```

Notice that in the above example, the `tf.data` API is used, but only in the sequential layer.
This is because `tf.data` has no concept of the random access layer.  As a result:

  - `tf.data.Dataset.shuffle()` can only approximate a shuffle.  Calling `.shuffle(N)` will read
    `N` records into a buffer and choose samples randomly from **those `N` records**, while a true
    shuffle chooses samples randomly from the **entire dataset**.  This shortcoming forces you
    to choose between memory footprint and the quality of your shuffle.  The only true
    shuffle with tf.data.Dataset.shuffle() is to read the entire dataset into memory.
  - `tf.data.Dataset.skip(N)` is as inefficient as possible.  Each of the `N` skipped records will
    still be read from disk and processed normally, according to all of the operations preceeding
    the `.skip()` call, making `.skip()` prohibitively expensive for most use cases.
  - Pausing and continuing training is only possible by saving the state of a `tf.data.Iterator`.
    However, saving a `tf.data.Iterator` does not work with all datasets.  In particular, it does
    not work with datasets created using `from_generator()`, which is the easiest way to create a
    `tf.data.Dataset`.

We have seen countless instances where `tf.data.Dataset` shortcomings have made life harder for
deep learning practitioners, so we set out to build something better.  We set out to build a new
data layer which could augment an existing `tf.data.Dataset` data loader with the properties should
come standard with every data loader.

At the same time, we wanted this new data layer to relieve another key pain point: high-performance
dataset caching and dataset versioning.

## What is `yogadl`?

We designed `yogadl` to be two things: a standalone caching layer to imbue existing data loaders
with the properties that come from a random-access layer, and a better interface for defining data
loaders in general.

### A standalone caching tool

Since `tf.data.Dataset`-based datasets have no random-access layer, `yogadl` caches them to disk in
a random-access-friendly way.  The storage mechanism is, in fact, nearly identical to how
[TensorPack caches datasets to disk](https://tensorpack.readthedocs.io/modules/dataflow.html#tensorpack.dataflow.LMDBSerializer),
only with some additional abstractions to allow dataset versioning, cloud storage, and all of the
wonderful features that a data loader with a random-access layer ought to have.

What does all this do for you?  A few things:

 - **Better training**: A `yogadl`-cached `tf.data.Dataset` will have better shuffling than a
   native `tf.data.Dataset`.  Additionally, pausing and continuing training mid-epoch will be
   simple and robust, and efficient sharding for distributed training comes standard.
 - **Faster data loading**: Slow data loader?  Don't waste your time optimizing it.  `yogadl` will
   save it in a high-performance cache the first time it is used, and all future uses will be
   fast and efficient.
 - **API-transparent**: Not all operations in the data loader are cacheable.  Data augmentation
   must be done at run time.  `yogadl` allows you to keep your existing data augmentation code.

### A better interface

At the core of `yogadl` is the `DataRef` interface, which creates an explicit boundary between the
random-access layer and the sequential layer.

We are not the first people to think of this: PyTorch separates the `DataSet` (the random-access
layer) from the `Sampler` (which defines the sequential layer).  Keras has a `Sequence` object
which defines the random-access layer, leaving the order of access (the sequential layer) to be
decided by the arguments to `model.fit()`.  Both `DataSet` and `Sequence` are already 100%
compatible with `yogadl`'s `DataRef` interface (although `yogadl` does not yet include those
adapters).

And yet, the world is still full of data loaders which are lacking.  At Determined AI, we are
dedicated to advancing the state of the art for training Deep Learning models, and we believe that
a better interface for data loading is a critical piece of that goal.  Any data loader which
implements the `DataRef` interface is capable of proper shuffling, pausing and continuing training
mid-epoch, and efficient multi-machine distributed training.

## What is `yogadl` _not_?

`yogadl` is not a data manipulation API.
[This](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
[world](https://tensorpack.readthedocs.io/tutorial/dataflow.html)
[has](https://keras.io/preprocessing/image/)
[more](https://pytorch.org/docs/stable/torchvision/ops.html)
[than](https://numpy.org/)
[enough](https://pandas.pydata.org/)
[of](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)
[those](https://opencv-python-tutroals.readthedocs.io/en/latest/).
Instead, `yogadl` seeks to be API-transparent so that you can continue to use your existing data
loading code, but with all the benefits of a high-performance, random-access cache.  If you have
data augmentation steps which cannot be cached, that code should continue to work without any
modifications.

`yogadl` does not (at this time) work with any data frameworks other than `tf.data.Dataset.`
First-class support for (tf.)Keras `Sequence` objects, PyTorch `DataSet` objects, and TensorPack
`DataFlow` objects is on the near-term roadmap.

`yogadl` offers basic dataset versioning, but it is not (at this time) a full-blown version control
for datasets.  Offering something like version control for datasets is on the roadmap as well.

<!-- ## How do I use `yogadl`? -->

<!-- TODO: code examples here. -->
