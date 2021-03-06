import tensorflow as tf
import numpy as np
from metaopt import MetaVAE
from data import get_image_paths_by_directories, load_image
import argparse

def model_fn(features, labels, mode, params, config):
    inputs = features
    inputs.set_shape((params.outer_batch_size, *inputs.shape[1:]))
    inputs = tf.stop_gradient(inputs)

    model = MetaVAE(num_inner_loops=params.num_inner_loops, first_order=params.first_order, adjust_loss=False)

    loss = model.get_loss(inputs[:, :params.inner_train_size], inputs[:, params.inner_train_size:])

    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = tf.train.cosine_decay(params.outer_learning_rate, tf.train.get_global_step(), 5000 * 150, params.outer_learning_rate / 10)
        opt = tf.train.AdamOptimizer(lr)
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

def get_input_fn(path, batch_size, images_per_batch, steps=100000000):
    print("Loading images from", path)
    image_shape = (32, 32, 1)
    image_paths_by_label = get_image_paths_by_directories(path, images_per_batch, target_size=image_shape)
    num_labels = len(image_paths_by_label)
    print("Loaded", num_labels, "labels and a total of", sum([len(im_paths) for im_paths in image_paths_by_label]), "images")
    print("Image shape:", image_shape)

    def _get_batch():
        chosen_label = np.random.randint(num_labels)
        label_image_paths = image_paths_by_label[chosen_label]
        #chosen_indices = np.random.choice(np.arange(len(label_images)), size=images_per_batch, replace=False)
        #chosen_indices = np.arange(images_per_batch)
        chosen_indices = np.random.choice(np.arange(images_per_batch), size=images_per_batch, replace=False)
        image_paths = [label_image_paths[index] for index in chosen_indices]
        images = np.array([load_image((path, image_shape)) for path in image_paths])

        if len(images.shape) == 3:
            images = np.expand_dims(images, -1)

        return images

    def input_fn():
        def f(d):
            ff = tf.py_func(_get_batch, [], tf.float32)
            ff.set_shape((images_per_batch, *image_shape))
            return ff

        dummy = tf.constant(0, shape=(steps,))
        dataset = tf.data.Dataset.from_tensor_slices(dummy)
        dataset = dataset.map(f)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(8)

        print("Dataset:", dataset)
        return dataset

    return input_fn

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-ob", "--outer_batch_size", type=int, default=32, help="Outer batch size")
    args.add_argument("-ib", "--inner_batch_size", type=int, default=8, help="Inner batch size")
    args.add_argument("-it", "--inner_train_size", type=int, default=5, help="Inner train batch size (must be smaller less inner_batch_size)")
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

    train_input_fn = get_input_fn(params.dataset_path, batch_size=params.outer_batch_size, images_per_batch=params.inner_batch_size)

    if params.dataset_test_path is None:
        estimator.train(train_input_fn)
    else:
        test_input_fn = get_input_fn(params.dataset_test_path, batch_size=params.outer_batch_size, images_per_batch=params.inner_batch_size)
        train_spec = tf.estimator.TrainSpec(train_input_fn)
        eval_spec = tf.estimator.EvalSpec(test_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
