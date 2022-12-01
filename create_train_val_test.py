import argparse
from os import listdir
from sklearn.model_selection import train_test_split
import pandas as pd
from bs4 import BeautifulSoup as bs
from skimage import io
import numpy as np


'''
Parses XML file and returns dataframe containing bounding box information
Input:
   xml: XML file to be parsed
Returns:
   df: Dataframe containing bounding box data
'''
def parse_xml(xml):
  label = xml.find_all("name")
  xmin = xml.find_all("xmin")
  ymin = xml.find_all("ymin")
  xmax = xml.find_all("xmax")
  ymax = xml.find_all("ymax")
  for i in range(len(label)):
      label[i] = label[i].text
      xmin[i] = xmin[i].text
      ymin[i] = ymin[i].text
      xmax[i] = xmax[i].text
      ymax[i] = ymax[i].text
  df = pd.DataFrame({"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
  return df

'''
Condenses labels for bounding box data
Inputs:
   df: Dataframe containing bounding box information
'''
def clean_up_data(df):
   cat1 = 'adult male'
   cat2 = 'male juvenile / female juvenile / adult female'
   cat3 = 'pup'

   cat1_diffs = ["Adult Male", "Adult male", "adult male seal"]

   cat2_diffs = ['juvenile male / juvenile female / adult female',
                 'male juvenile / female juvenile /adult female',
                 'Adult Female or Young Male',
                 'male juvenile/female juvenile/adult female',
                 'Juvenile male / juvenile female / adult female',
                 'Juvenile male/ juvenile female/ adult female',
                 'male juvenile / female juvenile / adulf female',
                 'adult female seal', '\\\\', ']', '\'']
   cat3_diffs = ['Pup', 'Juvenile', 'baby seal', 'juvenile seal']
   for cat1_diff in cat1_diffs:
       df.loc[df["label"] == cat1_diff, ['label']] = cat1
   for cat2_diff in cat2_diffs:
       df.loc[df["label"] == cat2_diff, ['label']] = cat2
   for cat3_diff in cat3_diffs:
       df.loc[df["label"] == cat3_diff, ['label']] = cat3
   return df

'''
Gets bounding box data from specified xml files
Inputs:
   in_path: Path to directory containing XML files
   xml: Array of names of XML files
Returns:
   df: Dataframe containing all bounding box data for XML files
'''
def get_bb(in_path, xml):
   df = pd.DataFrame()
   i = 0
   for x in xml:
      f = open(in_path + x)
      xml_file = bs("".join(f.readlines()), "lxml")
      df_temp = parse_xml(xml_file)
      df_temp.insert(0, "file_num", str(i).zfill(4))
      df = pd.concat([df, df_temp])
      f.close()
      i+=1
   df = clean_up_data(df)
   return df

def find_bounding_boxes(df_img, x_size, y_size, xmin, ymin, xmax, ymax):
  col_names = list(df_img.columns)
  col_names.append("percent_seal")
  df_bb = pd.DataFrame()
  for row in df_img.itertuples():
    bb_xmin = int(row[3])
    bb_ymin = int(row[4])
    bb_xmax = int(row[5])
    bb_ymax = int(row[6])

    bb_xmin_ = max(bb_xmin - xmin, 0)
    bb_ymin_ = max(bb_ymin - ymin, 0)
    bb_xmax_ = min(bb_xmax - xmin, xmax - xmin)
    bb_ymax_ = min(bb_ymax - ymin, ymax - ymin)
    bb_row = [row[1], row[2], bb_xmin_, bb_ymin_, bb_xmax_, bb_ymax_]

    # if a bounding box was found
    if ((bb_xmin_ >= 0 and bb_xmin_ <= x_size) and
       (bb_ymin_ >= 0 and bb_ymin_ <= y_size) and
       (bb_xmax_ >= 0 and bb_xmax_ <= x_size) and
       (bb_ymax_ >= 0 and bb_ymax_ <= y_size)):
      height = bb_ymax_ - bb_ymin_; length = bb_xmax_ - bb_xmin_
      area = height * length # area of bb in image
      bb_area = (bb_xmax - bb_xmin) * (bb_ymax - bb_ymin) # total area of bb

      if (bb_area == 0): # ignore bounding boxes that don't have any area
        break

      percent_seal_present = area/bb_area # % of bb present in subimage
      # print("original bb coordinates:", row)
      # print("subimage coordinates:", [xmin, ymin, xmax, ymax])
      # print("new bb coordinates:", bb_row)
      # print(percent_seal_present, area, bb_area)
      bb_row.append(percent_seal_present)
      df_bb = df_bb.append(pd.Series(bb_row, index=col_names), ignore_index=True)

  if len(df_bb) == 0:
    df_bb = None
  return df_bb

def split_image(img, num_subimgs, df_img, x_size, y_size, x_int, y_int):
    # _size is the length in each direction
    # _int is the interval you shift right or down by
    # ex. (150, 150, 75, 75) - each split creates a 150x150 images and shifts over by 75 pixels each run through.
    x_len = img.shape[1]
    y_len = img.shape[0]

    bb_data = {}
    sub_images = []
    num_image = num_subimgs
    # crops like reading a book
    i = 0
    while (i < y_len):
        # updates the new y coordinates
        y1 = i
        if i + y_size > y_len:
            y1 = y_len-y_size
            y2 = y_len
            i = y_len
        else:
            y2 = i + y_size
            i += y_int

        j = 0
        while (j < x_len):
            # updates the new x coordinates
            x1 = j
            if j + x_size > x_len:
                x1 = x_len - x_size
                x2 = x_len
                j = x_len
            else:
                x2 = j + x_size
                j += x_int
            df_bb = find_bounding_boxes(df_img, x_size, y_size, x1, y1, x2, y2)
            bb_data[num_image] = df_bb
            cropped = img[y1:y2,x1:x2]
            sub_images.append(cropped)
            num_image += 1

    bb_data = pd.Series(bb_data)
    return sub_images, bb_data


'''
Split images into subimages and gets respective bounding box data
Inputs: 
   x_size: X size for sub image
   y_size: Y size for sub image
   x_int: X size for step 
   y_int: Y size for step
   imgs: Array of images to be split
   df: DataFrame containing bounding box data for respective images
Returns:
   all_sub_imgs: Array of sub images
   all_bb_data: Array of data frames containing bounding box data for each subimage, None if image has no bounding box
'''
def split_and_label(x_size, y_size, x_int, y_int, imgs, df):
  num_subimgs = 0
  all_sub_imgs = []
  all_bb_data = pd.Series(dtype='object')
  for i in range(len(imgs)):
      fnum = str(i).zfill(4)
      df_img = df[df["file_num"] == fnum]
      sub_imgs, bb_data = split_image(imgs[i], num_subimgs, df_img, x_size, y_size, x_int, y_int)
      all_sub_imgs += sub_imgs
      all_bb_data = pd.concat([all_bb_data, bb_data])
      num_subimgs += len(sub_imgs)
  return all_sub_imgs, all_bb_data


'''
Splits into sub images and bounding boxes and writes to specified path
Inputs:
   dim: Tuple of sub image size and step for sub images (x_size, y_size, x_step, y_step)
   img_path: Path to directory containing XML and image data
   imgs: array of images to split into subimages
   xml: Dataframe contaning bounding box information
   name: Name of the file when written
   out_path: Path to output directory
'''
def split_label_write(dim, img_path, imgs, xml, name, out_path):
   bb = get_bb(img_path, xml)
   sub_imgs, bb_data = split_and_label(dim[0], dim[1], dim[2], dim[3], imgs, bb)
   print("Legnth of", name + ":", len(sub_imgs))
   np.save(out_path + name + "_images.npy", sub_imgs)
   np.save(out_path + name + "_bb_data.npy", bb_data)

def main():
   # Defualt value initialization
   image_data_path = "/data2/noses_data/image_data/"
   out_path = "/data2/noses_data/cnn_data/"
   
   # Get proportions for test, train, and validation split
   parser = argparse.ArgumentParser(description = "Split for test, training and validation sets")
   parser.add_argument("-t", required = True, help= "Test train split proportion")
   parser.add_argument("-v", required = True, help = "Train validation split ratio")
   parser.add_argument("-i", required = False, help = "Path to image data and xml files")
   parser.add_argument("-o", required = False, help = "Path to output directory")
   args = parser.parse_args()
   test_train_ratio = float(args.t)
   train_val_ratio = float(args.v)
   if args.i is not None:
      image_data_path = args.i
   if args.o is not None:
      out_path = args.o
   
   # Read in images and xml filesa
   files = listdir(image_data_path)
   xml_files = []
   imgs = []
   for x in files:
      if ".xml" in x:
         xml_files.append(x)
      else:
         img = io.imread(image_data_path + x, plugin="matplotlib")
         imgs.append(img)
   
   # Create Splits
   train_xml, test_xml, train_img, test_img = train_test_split(xml_files, imgs, test_size = 1 - test_train_ratio, random_state = 5)
   train_xml, val_xml, train_img, val_img = train_test_split(train_xml, train_img, test_size = 1 - train_val_ratio, random_state = 5)

   # Split images and write to file
   dim = (150, 150, 75, 75)
   # For Train
   split_label_write(dim, image_data_path, train_img, train_xml, "train", out_path)
   # For Val
   split_label_write(dim, image_data_path, val_img, val_xml, "val", out_path)
   # For Test
   split_label_write(dim, image_data_path, test_img, test_xml, "test", out_path)

if __name__ == '__main__':
   main()
