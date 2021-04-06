import sys
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0


def main():
    output_dir = sys.argv[1]
    base_model = EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )
    # model = tf.keras.Model(inputs=base_model.input, outputs=x)
    x = base_model.get_layer('block7a_project_bn').output
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    tf.saved_model.save(model, output_dir)


if __name__ == '__main__':
    main()
