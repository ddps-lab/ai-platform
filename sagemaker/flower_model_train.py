import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras

import argparse
import os
import json

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--img_height', type=int, default=180)
    parser.add_argument('--img_width', type=int, default=180)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_ds = tf.data.Dataset.load(os.path.join(args.train,'train'))
    val_ds = tf.data.Dataset.load(os.path.join(args.train, 'test'))

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(args.img_height,
                                        args.img_width,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # 모델 만들기
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(5, name="outputs")
    ])

    #모델 컴파일
   model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.Precision(name='precision'), 
                       tf.keras.metrics.Recall(name='recall'), 
                       tf.keras.metrics.F1Score(name='f1_score')])

    # 모델 훈련
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=2
    )

    if args.current_host == args.hosts[0]:
        model.save(os.path.join(args.sm_model_dir, '000000001'), 'flower_model.keras')
