import sys
sys.path.append("../../")

import tensorflow as tf
import numpy as np
from metaclassification import MetaConvClassifier
from data import load_images_by_directories
import argparse

def model_fn(features, labels, mode, params, config):
    inputs = features
    inputs.set_shape((params.outer_batch_size, *inputs.shape[1:]))
    labels.set_shape((params.outer_batch_size, *labels.shape[1:]))

    inputs = tf.stop_gradient(inputs)
    labels = tf.stop_gradient(labels)

    model = MetaConvClassifier(num_ways=params.num_ways, num_inner_loops=params.num_inner_loops, first_order=params.first_order, adjust_loss=False)

    train_inputs, train_labels = inputs[:, :params.num_ways*params.inner_train_size], labels[:, :params.num_ways*params.inner_train_size]
    test_inputs, test_labels = inputs[:, params.num_ways*params.inner_train_size:], labels[:, params.num_ways*params.inner_train_size:]

    loss = model.get_loss(train_inputs, train_labels, test_inputs, test_labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdamOptimizer(params.outer_learning_rate)
        gvs = opt.compute_gradients(loss)
        gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
        train_op = opt.apply_gradients(gvs, global_step=tf.train.get_global_step())
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=None,
        loss=loss,
        train_op=train_op,
    )

def get_input_fn(path, batch_size, num_ways, images_per_way, steps=100000000):
    print("Loading images from", path)
    images_by_label = load_images_by_directories(path, images_per_way, target_size=(28, 28, 3))
    image_shape = images_by_label[0][0].shape
    num_labels = len(images_by_label)
    print("Loaded", num_labels, "labels and a total of", sum([len(im) for im in images_by_label]), "images")
    print("Image shape:", image_shape)
    print("Min / max:", np.min(images_by_label[0]), np.max(images_by_label[0]))

    def _get_batch():
        chosen_labels = np.random.choice(np.arange(num_labels), size=num_ways)
        images = []
        labels = []

        for label_index, label in enumerate(chosen_labels):
            label_images = images_by_label[label]
            
            chosen_indices = np.random.choice(np.arange(len(label_images)), size=images_per_way, replace=False)
            images.append(label_images[chosen_indices])
            labels.append(np.array([label_index] * images_per_way, dtype=np.int32))

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        # 111222333 -> 123123123 (so last shots can be taken as test and have all ways)
        images = images.reshape(num_ways, images_per_way, *image_shape)
        labels = labels.reshape(num_ways, images_per_way)
        images = images.transpose(1, 0, *range(2, 2 + len(image_shape)))
        labels = labels.transpose(1, 0)
        images = images.reshape(num_ways * images_per_way, *image_shape)
        labels = labels.reshape(num_ways * images_per_way)

        return images, labels

    def input_fn():
        def f(d):
            images, labels = tf.py_func(_get_batch, [], (tf.float32, tf.int32))
            images.set_shape((num_ways * images_per_way, *image_shape))
            labels.set_shape((num_ways * images_per_way,))
            return images, labels

        dummy = tf.constant(0, shape=(steps,))
        dataset = tf.data.Dataset.from_tensor_slices(dummy)
        dataset = dataset.map(f)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)

        print("Dataset:", dataset)
        return dataset

    return input_fn

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-ob", "--outer_batch_size", type=int, default=32, help="Outer batch size")
    args.add_argument("-nw", "--num_ways", type=int, default=5, help="Number of ways (labels per classification task)")
    args.add_argument("-ib", "--inner_batch_size", type=int, default=2, help="Inner batch size")
    args.add_argument("-it", "--inner_train_size", type=int, default=1, help="Inner train batch size (must be smaller less inner_batch_size)")
    args.add_argument("-d", "--dataset_path", type=str, default="omniglot/train", help="Path to dataset (images in same folder will be treated as same label)")
    args.add_argument("-dt", "--dataset_test_path", type=str, default=None, help="Path to test dataset (images in same folder will be treated as same label). No testing is done if None.")
    args.add_argument("-m", "--model_path", type=str, default="models", help="Estimator model path. Will load existing models. Also saves tensorboard summaries to the same directory.")
    args.add_argument("-il", "--num_inner_loops", type=int, default=5, help="Number of inner network optimization steps.")
    args.add_argument("-olr", "--outer_learning_rate", type=float, default=0.001, help="Learning rate for the outer network.")
    args.add_argument("-f", "--first_order", dest="first_order", action="store_true", help="Use first order approximation for outer gradients.")
    args.set_defaults(first_order=False)
    params = args.parse_args()

    print("Params:", params)

    tf.logging.set_verbosity(tf.logging.INFO)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True

    run_config = tf.estimator.RunConfig(
        model_dir=params.model_path,
        save_summary_steps=50,
        session_config=session_config,
    )

    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=run_config)

    train_input_fn = get_input_fn(params.dataset_path, batch_size=params.outer_batch_size, num_ways=params.num_ways, images_per_way=params.inner_batch_size)

    if params.dataset_test_path is None:
        estimator.train(train_input_fn)
    else:
        test_input_fn = get_input_fn(params.dataset_test_path, batch_size=params.outer_batch_size, num_ways=params.num_ways, images_per_way=params.inner_batch_size)
        train_spec = tf.estimator.TrainSpec(train_input_fn)
        eval_spec = tf.estimator.EvalSpec(test_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
