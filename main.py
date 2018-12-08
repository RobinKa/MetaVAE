import tensorflow as tf
import numpy as np
from models import MetaVAE
from data import load_images_by_directories

def model_fn(features, labels, mode, params, config):
    print("Features:", features)
    image = features
    #image.set_shape([32, 6, 28, 28, 1])

    print("Image:", image)

    model = MetaVAE(num_inner_loops=1)
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op = model.get_train(image)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=None,
        loss=loss,
        train_op=train_op,
    )

def get_input_fn(path, batch_size, images_per_batch, steps):
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images_by_label = [np.array([image for j, image in enumerate(train_images) if train_labels[j] == i]) for i in range(10)]
    image_shape = train_images[0].shape
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    """

    print("Loading images from", path)
    images_by_label = load_images_by_directories(path, images_per_batch)
    image_shape = images_by_label[0][0].shape
    num_labels = len(images_by_label)
    print("Loaded", num_labels, "labels and a total of", sum([len(im) for im in images_by_label]), "images")
    print("Image shape:", image_shape)

    def _get_batch():
        chosen_label = np.random.randint(num_labels)
        label_images = images_by_label[chosen_label]
        chosen_indices = np.random.choice(np.arange(len(label_images)), size=images_per_batch, replace=False)
        images = (label_images[chosen_indices] / 255.).astype(np.float32)
        return np.expand_dims(images, -1)

    def input_fn():
        def f(d):
            ff = tf.py_func(_get_batch, [], tf.float32)
            ff.set_shape((images_per_batch, *image_shape, 1))
            return ff

        # [ImagesPerBatch, *ImageShape]
        dummy = tf.constant(0, shape=(steps,))
        dataset = tf.data.Dataset.from_tensor_slices(dummy)
        dataset = dataset.map(f)
        # [BatchSize, ImagesPerBatch, *ImageShape]
        #dataset = dataset.repeat(steps)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)

        print("Dataset:", dataset)

        return dataset
    return input_fn

def main():
    tf.enable_eager_execution()
    tf.executing_eagerly() 

    tf.logging.set_verbosity(tf.logging.INFO)

    run_config = tf.estimator.RunConfig(
        model_dir="models",
    )

    train_input_fn = get_input_fn("omniglot/train", batch_size=32, images_per_batch=8, steps=10000)
    test_input_fn = get_input_fn("omniglot/test", batch_size=32, images_per_batch=8, steps=10000)

    estimator = tf.estimator.Estimator(model_fn=model_fn, params={}, config=run_config)

    train_spec = tf.estimator.TrainSpec(train_input_fn)
    eval_spec = tf.estimator.EvalSpec(test_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
