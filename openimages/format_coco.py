import os
import pandas as pd
import mmcv
from tqdm import tqdm

def create_coco_json(annotation_file, out_dir=None):
    """
    Creates two COCO json files from the the MIAP dataset. One contains the annotations about gender, the other about age.
    The script saves the output files in the same directory of the annotation file, unless specified otherwise by the out_file variable.
    args:
        annotation_file: str, Path to the annotation file. This script assumes that the images are in the same folder as the annotation file.
        out_dir: str, Path to the output file.
    """

    dataset = annotation_file.split('_')[-1][:-4] # train, val or test
    base_dir = os.path.dirname(annotation_file)
    out_dir = base_dir if out_dir is None else out_dir
    out_file_gender = os.path.join(out_dir, f"miap_{dataset}_gender.json")
    out_file_age = os.path.join(out_dir, f"miap_{dataset}_age.json")
    df = pd.read_csv(annotation_file)

    # make a list with the unique ImageId's
    unique_image_ids = df['ImageID'].unique()

    gender_annotations = []
    age_annotations = []
    images = []
    obj_count = 0

    for idx, filename in enumerate(tqdm(unique_image_ids, desc=f"Working on {dataset} set...")):
        tmp_df = df[df['ImageID'] == filename]
        img_path = os.path.join(base_dir, dataset, f'{filename}.jpg')
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))

        # Loop over the (potential) possible bounding boxes
        for _, bounding_box_row in tmp_df.iterrows():

            # Extract BB coordinates
            x_min, x_max, y_min, y_max = bounding_box_row['XMin'], bounding_box_row['XMax'], bounding_box_row['YMin'], bounding_box_row['YMax']

            # Data annotation for gender 
            data_annotation_gender = dict(
                    id = obj_count,
                    image_id = idx,
                    category_id = get_category_id(bounding_box_row['GenderPresentation'], feature="gender"),
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min],
                    area = (x_max - x_min) * (y_max - y_min),
                    iscrowd = int(len(tmp_df) > 1)
                    )
                    
            # Data annotation for age 
            data_annotation_age = dict(
                    id = obj_count,
                    image_id = idx,
                    category_id = get_category_id(bounding_box_row['AgePresentation'], feature="age"),
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min],
                    area = (x_max - x_min) * (y_max - y_min),
                    iscrowd = int(len(tmp_df) > 1)
                    )

            # Append the annotations to the lists
            gender_annotations.append(data_annotation_gender)
            age_annotations.append(data_annotation_age)
            obj_count += 1


    categories=[
            {'supercategory': 'gender', 'id':0, 'name': 'Predominantly Feminine'},
            {'supercategory': 'gender', 'id':1, 'name': 'Unknown'},
            {'supercategory': 'gender', 'id':2, 'name': 'Predominantly Masculine'},
            {'supercategory': 'age',    'id':3, 'name': 'Unknown'},
            {'supercategory': 'age',    'id':4, 'name': 'Young'},
            {'supercategory': 'age',    'id':5, 'name': 'Middle'},
            {'supercategory': 'age',    'id':6, 'name': 'Older'}
            ]

    coco_format_gender = dict(
        images=images,
        annotations=gender_annotations,
        categories=categories
        )
    
    coco_format_age = dict(
        images=images,
        annotations=age_annotations,
        categories=categories
        )

    # Save the COCO json files
    mmcv.dump(coco_format_gender, out_file_gender)
    mmcv.dump(coco_format_age, out_file_age)


def get_category_id(string, feature):
    """
    Convert a string to a category id. More specifically, if feature is "gender":
        Predominantly Feminine  -> 0
        Uknown                  -> 1
        Predominantly Masculine -> 2
    If feature is "age":
        Uknown                  -> 3
        Young                   -> 4
        Middle                  -> 5
        Older                   -> 6
    args:
        string: the string to convert
        feature: a string, either "gender" or "age"
    returns:
        a category id (int)
    """

    assert feature in ["gender", "age"], f"Unknown feature {feature}"

    if feature == "gender":
        if string == "Predominantly Feminine":
            return 0
        elif string == "Unknown":
            return 1
        elif string == "Predominantly Masculine":
            return 2
        else:
            raise ValueError(f"Unknown string {string}")
    elif feature == "age":
        if string == "Unknown":
            return 3
        elif string == "Young":
            return 4
        if string == "Middle":
            return 5
        elif string == "Older":
            return 6
        else:
            raise ValueError(f"Unknown string {string}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help='Path to the annotation file')
    parser.add_argument('--out_dir', type=str, default=None, help='Path to the output file')
    args = parser.parse_args()

    create_coco_json(args.filepath, args.out_dir)

