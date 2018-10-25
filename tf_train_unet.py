import argparse
import tf_unet_model
import bce_dice_loss
import tf_img_prepro_aug


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--images_path", type = str  )
parser.add_argument("--labels_path", type = str  )
parser.add_argument("--input_height", type=int , default = 256  )
parser.add_argument("--input_width", type=int , default = 256 )

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights_path", type = str , default = "")

args = parser.parse_args()

images_path = args.images_path
labels_path = args.labels_path
batch_size = args.batch_size
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights_path = args.load_weights_path

if validate:
    
 
img_data = tf.data.Dataset.list_files(images_path)
img_data = img_data.shuffle(500)
img_data = img_data.map(lambda x: load_jpeg(x))
img_data = img_data.batch(1)
labels_data = tf.data.Dataset.list_files(labels_path)
labels_data = labels_data.shuffle(500)
labels_data = labels_data.map(lambda x: load_jpeg(x))
labels_data = labels_data.batch(1)
model = tf_unet_model()
