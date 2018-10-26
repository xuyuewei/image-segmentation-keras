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



num_of_samples = len(img_array)
img_array = tf.data.Dataset.list_files(images_path,shuffle=False)
labels_array = tf.data.Dataset.list_files(labels_path,shuffle=False)
img_labels = tf.data.Dataset.from_tensor_slices((img_array,labels_array))
img_labels = img_labels.shuffle(num_of_samples)

#train_val_split
val_labels_data = None
if validate:
    val_array = img_labels[:num_of_samples//10]
    img_labels = img_labels[num_of_samples//10:]
    val_data = val_array.map(lambda x: (load_jpeg(x[0])[:,:input_width//2],load_jpeg(x[0])[:,input_width//2:]))
    val_data = val_data.batch(1)
    
img_labels_data = img_labels.map(lambda x: (load_jpeg(x[0])[:,:input_width//2],load_jpeg(x[0])[:,input_width//2:]))    
img_labels_data = img_labels_data.batch(1)

#create unet model
model = tf_unet_model()
                                         





