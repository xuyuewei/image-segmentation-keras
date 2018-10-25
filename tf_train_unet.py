import argparse
import tf_unet_model
import bce_dice_loss
import tf_img_prepro_aug


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images_path", type = str  )
parser.add_argument("--train_segs_path", type = str  )
parser.add_argument("--input_height", type=int , default = 256  )
parser.add_argument("--input_width", type=int , default = 256 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images_path", type = str , default = "")
parser.add_argument("--val_segs_path", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights_path", type = str , default = "")

args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_segs_path
train_batch_size = args.batch_size
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights_path

if validate:
	val_images_path = args.val_images_path
	val_segs_path = args.val_segs_path
	val_batch_size = args.val_batch_size
  
model = tf_unet_model()
