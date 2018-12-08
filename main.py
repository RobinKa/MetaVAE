import tensorflow as tf
import numpy as np
from models import MetaVAE

def model_fn(features, labels, mode, params, config):
    image = features["image"]
    #image.set_shape([32, 6, 28, 28, 1])

    print("Image:", image)

    model = MetaVAE()
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

def get_input_fn(batch_size, images_per_batch, steps, augment):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images_by_label = [np.array([image for j, image in enumerate(train_images) if train_labels[j] == i]) for i in range(10)]
    image_shape = train_images[0].shape
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None

    def _get_batch():
        batch = []
        for i in range(batch_size):
            chosen_label = np.random.randint(10)
            label_images = train_images_by_label[chosen_label]
            chosen_indices = np.random.choice(np.arange(len(label_images)), size=images_per_batch, replace=False)
            images = (label_images[chosen_indices] / 255.).astype(np.float32)
            batch.append(images)
        return np.stack(batch)

    def input_fn():
        # [ImagesPerBatch, *ImageShape]
        batch = tf.py_func(_get_batch, [], tf.float32)
        batch.set_shape((batch_size, images_per_batch, *image_shape))
        batch = tf.expand_dims(batch, -1)

        dataset = tf.data.Dataset.from_tensors({"image": batch})
        dataset = dataset.repeat(steps)

        # [BatchSize, ImagesPerBatch, *ImageShape]
        #dataset = dataset.batch(batch_size)
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

    input_fn = get_input_fn(batch_size=32, images_per_batch=8, steps=10000, augment=True)

    estimator = tf.estimator.Estimator(model_fn=model_fn, params={}, config=run_config)
    estimator.train(input_fn)


if __name__ == "__main__":
    main()
