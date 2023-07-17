import tensorflow as tf
import tensorflow_datasets as tfds
import os

def build_model():
    """Constructs the ML model used to predict handwritten digits."""

    # this an example
    ## you should change it
    image = tf.keras.layers.Input(shape=(28, 28, 1))

    y = tf.keras.layers.Conv2D(filters=32,
                                kernel_size=5,
                                padding='same',
                                activation='relu')(image)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same')(y)
    y = tf.keras.layers.Conv2D(filters=32,
                                kernel_size=5,
                                padding='same',
                                activation='relu')(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same')(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(1024, activation='relu')(y)
    y = tf.keras.layers.Dropout(0.4)(y)

    probs = tf.keras.layers.Dense(10, activation='softmax')(y)

    model = tf.keras.models.Model(image, probs, name='mnist')

    return model



@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image(example, feature):
    """Convert image to float32 and normalize from [0, 255] to [0.0, 1.0]."""
    return tf.cast(feature.decode_example(example), dtype=tf.float32) / 255



def run(flags_obj, datasets_override=None, strategy_override=None):
    """Run MNIST model training and eval loop using native Keras APIs.

    Args:
        flags_obj: An object containing parsed flag values.
        datasets_override: A pair of `tf.data.Dataset` objects to train the model,
                        representing the train and test sets.
        strategy_override: A `tf.distribute.Strategy` object to use for model.

    Returns:
        Dictionary of training and eval stats.
    """
    
    ######################################################################################################
    ######################## this is data ##################################
    mnist = tfds.builder('mnist', data_dir=flags_obj.data_dir)
    
    if flags_obj.download:
        mnist.download_and_prepare()

    mnist_train, mnist_test = datasets_override or mnist.as_dataset(
        split=['train', 'test'],
        decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
        as_supervised=True)
    
    train_input_dataset = mnist_train.cache().repeat().shuffle(buffer_size=50000).batch(flags_obj.batch_size)
    eval_input_dataset = mnist_test.cache().repeat().batch(flags_obj.batch_size)
    ###################### above should be changed based on the application ###############################
    #######################################################################################################
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( 0.05, decay_steps=100000, decay_rate=0.96)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    model = build_model()
    
    ## this should be changed for specific application
    ######### ////////////////////\\\\\\\\\\\\\\\\\\\\ #########################
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    num_train_examples = mnist.info.splits['train'].num_examples
    train_steps = num_train_examples // flags_obj.batch_size
    train_epochs = flags_obj.train_epochs

    
    num_eval_examples = mnist.info.splits['test'].num_examples
    num_eval_steps = num_eval_examples // flags_obj.batch_size
    ########## \\\\\\\\\\\\\\\\\\\\\////////////////////#########################

    history = model.fit(
        train_input_dataset,
        epochs=train_epochs,
        steps_per_epoch=train_steps,
        validation_steps=num_eval_steps,
        validation_data=eval_input_dataset)

    export_path = os.path.join(flags_obj.model_dir, 'saved_model')
    model.save(export_path, include_optimizer=False)

    return 'OK'





from absl import flags
import absl.logging as logging
FLAGS = flags.FLAGS

def define_mnist_flags():
    """Define command line flags for MNIST model."""
    flags.DEFINE_bool('download', True,
                        'Whether to download data to `--data_dir`.')
    flags.DEFINE_integer('profiler_port', 9012,
                        'Port to start profiler server on.')
    flags.DEFINE_integer('batch_size', 1024, 'The batch size for training')
    flags.DEFINE_string('model_dir', default=None, help='The directory where the model will be saved')
    flags.DEFINE_string('data_dir', default=None, help='The directory where the input data is stored')
    flags.DEFINE_integer('train_epochs', default=10, help='The number of epochs to train the model')



def main(_):
  stats = run(flags.FLAGS)
  logging.info('Run status:\n%s', stats)

from absl import app
if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_mnist_flags()
  app.run(main)
