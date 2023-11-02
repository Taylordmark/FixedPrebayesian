import argparse

import tensorflow as tf
import numpy as np

from utils.coco_dataset_manager import *

from utils.custom_retinanet import *


parser = argparse.ArgumentParser(description="Model Trainer")

parser.add_argument("--json_path", "-j", type=str, help="Path of the coco annotation used to download the dataset")
parser.add_argument("--save_path", "-s", type=str, help="Path to save \ load the downloaded dataset")
parser.add_argument("--download", "-d", type=str, help="Whether to download the dataset images or not", default="True")
parser.add_argument("--batch_size", "-b", type=int, default=4)
parser.add_argument("--epochs", "-e", help="number of epochs", default=50, type=int)
parser.add_argument("--num_imgs", "-n", help="number of images", default=50, type=int)
parser.add_argument("--checkpoint_path", "-p", help="path to save checkpoint", default="retinanet")
parser.add_argument("--mode", "-m", help="enter train, test, or traintest to do both", default="traintest", type=str)
parser.add_argument("--max_iou", "-i", help="max iou", default=.5, type=float)
parser.add_argument("--min_confidence", "-c", help="min confidence", default=.01, type=float)



args = parser.parse_args()

model_dir = args.checkpoint_path
label_encoder = LabelEncoder()

num_classes = 80
batch_size = args.batch_size

b_mod = int(args.epochs/batch_size)
learning_rates = [0.005, 0.01, .05, .01, .005, .00025]
learning_rate_boundaries = [5*b_mod, 10*b_mod,25*b_mod, 50*b_mod, 100*b_mod]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

coco_ds = CocoDSManager(args.json_path, args.save_path, max_samples=args.num_imgs, download=args.download == "True")


train_dataset = coco_ds.train_ds
val_dataset = coco_ds.val_ds


resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone, args.min_confidence, args.max_iou)


if "train" in args.mode:
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)


    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    autotune = tf.data.AUTOTUNE

    train_dataset = format_dataset(train_dataset, autotune, label_encoder, batch_size, False)
    val_dataset = format_dataset(val_dataset, autotune, label_encoder, batch_size, False)

    # Uncomment the following lines, when training on full dataset
    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
    # val_steps_per_epoch = \
    #     dataset_info.splits["validation"].num_examples // batch_size

    # train_steps = 4 * 100000
    # epochs = train_steps // train_steps_per_epoch

    epochs = args.epochs

    # Running 100 training and 50 validation steps,
    # remove `.take` when training on the full dataset

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
)


# Change this to `model_dir` when not using the downloaded weights
if "test" in args.mode:

    print("TEST")

    latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)

    model.load_weights(latest_checkpoint)






    for sample in coco_ds.test_ds.take(3):


        print("TESTING")
        image = tf.cast(sample["image"], dtype=tf.float32)

    
        input_image, ratio = prepare_image(image)
        detections = model.predict(input_image)

        print(detections)

        boxes = np.asarray(detections["boxes"])

        print(boxes)
        cls_prob = np.asarray(detections["cls_prob"])

        cls_id = np.asarray(detections["cls_idx"])

        key_list = list(coco_ds.coco.cats.keys())
        cls_name = [coco_ds.coco.cats[key_list[x]] for x in cls_id[0]]



        print(cls_name)

        # cls_id = 

        # if len(cls_prob[0][0] > 0):
        #     cls_id = np.argmax(cls_prob, axis=1)


        visualize_detections(image, boxes[0], cls_name, cls_prob[0][0])
        #show_frame_no_deep(np.asarray(image), np.asarray(detections["boxes"][0]), 2000)
