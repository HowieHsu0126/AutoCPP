import random
import time
import argparse
import os.path as osp
from tqdm import tqdm

"""
Dataset | Observation Time           | Prediction Time               |
---------------------------------------------------------------------|
weibo   | 3600 (1 hour)              | 3600*24 (86400, 1 day)        |
twitter | 3600*24*2 (172800, 2 days) | 3600*24*32 (2764800, 32 days) |
acm     | 3 (years)                  | 10 (years)                    |
aps     | 365*3 (1095, 3 years)      | 365*20+5 (7305, 20 years)     |
dblp    | 5 (years)                  | 20 (years)                    |
"""

config = {
    "weibo": {
        "Observation Time": 3600,
        "Prediction Time": 86400
    },
    "twitter": {
        "Observation Time": 172800,
        "Prediction Time": 2764800
    },
    "acm": {
        "Observation Time": 3,
        "Prediction Time": 10
    },
    "aps": {
        "Observation Time": 1095,
        "Prediction Time": 7305
    },
    "dblp": {
        "Observation Time": 5,
        "Prediction Time": 20
    }
}


def process_and_write_data(filtered_data, cascades_type, observation_time, prediction_time,
                           file_full, file_train, file_val, file_test, file_unlabel, unlabel):
    # Separate data for training, validation and testing
    filtered_data_full = []
    filtered_data_train = []
    filtered_data_val = []
    filtered_data_test = []
    for line in filtered_data:
        cascade_id = line.split('\t')[0]
        filtered_data_full.append(line)
        if not unlabel:
            if cascades_type[cascade_id] == 0:
                filtered_data_train.append(line)
            elif cascades_type[cascade_id] == 1:
                filtered_data_val.append(line)
            elif cascades_type[cascade_id] == 2:
                filtered_data_test.append(line)
    # Define a function to write data to file

    def file_write(file_name, cascade_id, observation_path, label=None):
        if label is not None:
            file_name.write(cascade_id + '\t' +
                            '\t'.join(observation_path) + '\t' + label + '\n')
        else:
            file_name.write(cascade_id + '\t' +
                            '\t'.join(observation_path) + '\n')
    # Write labeled data to files
    if not unlabel:
        with open(file_full, 'w') as data_full, open(file_train, 'w') as data_train, open(file_val, 'w') as data_val, open(file_test, 'w') as data_test:
            for line in tqdm(filtered_data_full + filtered_data_train + filtered_data_val + filtered_data_test, desc='Writing'):
                parts = line.split('\t')
                cascade_id = parts[0]
                observation_path = []
                label = 0
                paths = parts[4].split(' ')
                for p in paths:
                    nodes = p.split(':')[0].split('/')
                    time_now = int(p.split(":")[1])
                    if time_now < observation_time:
                        observation_path.append(
                            ",".join(nodes) + ":" + str(time_now))
                    if time_now < prediction_time:
                        label += 1
                label = str(label - len(observation_path))
                file_write(data_full, cascade_id, observation_path, label)
                if cascades_type[cascade_id] == 0:
                    file_write(data_train, cascade_id, observation_path, label)
                elif cascades_type[cascade_id] == 1:
                    file_write(data_val, cascade_id, observation_path, label)
                elif cascades_type[cascade_id] == 2:
                    file_write(data_test, cascade_id, observation_path, label)
    # Write unlabeled data to files
    else:
        with open(file_unlabel, 'w') as data_unlabel:
            for line in tqdm(filtered_data, desc='Writing'):
                parts = line.split('\t')
                cascade_id = parts[0]
                observation_path = []
                paths = parts[4].split(' ')
                for p in paths:
                    nodes = p.split(':')[0].split('/')
                    time_now = int(p.split(":")[1])
                    if time_now < observation_time:
                        observation_path.append(
                            ",".join(nodes) + ":" + str(time_now))
                file_write(data_unlabel, cascade_id, observation_path)


def generate_cascades(dataset, observation_time, prediction_time, unlabel,
                      filename, file_full, file_train, file_val, file_test, file_unlabel, train_ratio=.5, seed=3407):
    # sourcery skip: low-code-quality
    """
    Generate cascades from the data in filename and save the labeled data in training, validation, and testing files,
    and the unlabeled data in an unlabeled file.

    Args:
        observation_time (int): The time of observation.
        prediction_time (int): The time of prediction.
        unlabel (bool): Flag to indicate whether to generate unlabeled data.
        filename (str): The path to the file containing the original data.
        file_train (str): The path to the file to save the training data.
        file_val (str): The path to the file to save the validation data.
        file_test (str): The path to the file to save the testing data.
        file_unlabel (str): The path to the file to save the unlabeled data.
        seed (int): Seed for random number generation.

    Returns:
        None
    """
    filtered_data = []
    with open(filename) as file:
        cascades_type = {}
        cascades_time_dict = {}
        cascade_total = 0
        cascade_valid_total = 0
        for line in tqdm(file, desc=f'Processing {filename}'):
            cascade_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]
            if not unlabel:
                conditions = [
                    ('weibo', 8, 18, float, '%H', '+8'),
                    ('twitter', 10, 4, float, '%m%d', None),
                    ('acm', '2006', None, str, None, None),
                    ('aps', '1997-12', None, str, None, None),
                    ('dblp', '1997', None, str, None, None)
                ]
                for cond in conditions:
                    if cond[0] in dataset and cond[1] > (cond[3](parts[2]) if cond[4] is None else int(time.strftime(cond[4], (time.localtime if cond[5] is None else time.gmtime)(cond[3](parts[2]))))):
                        continue
            else:
                conditions = [
                    ('weibo', 18, 24, float, '%H', '+8'),
                    ('twitter', 10, 3, float, '%m', None),
                    ('acm', '2006', None, str, None, None),
                    ('aps', '1997-12', None, str, None, None),
                    ('dblp', '1997', None, str, None, None)
                ]
                for cond in conditions:
                    if cond[0] in dataset and cond[1] <= (cond[3](parts[2]) if cond[4] is None else int(time.strftime(cond[4], (time.localtime if cond[5] is None else time.gmtime)(cond[3](parts[2]))))):
                        continue

            paths = parts[4].strip().split(' ')
            observation_path = []
            p_o = sum(int(p.split(':')[1]) < observation_time for p in paths)
            if (dataset == 'dblp' and p_o < 5) or (dataset != 'dblp' and p_o < 10):
                continue

            observation_path.sort(key=lambda tup: tup[1])
            cascades_time_dict[cascade_id] = 0 if 'aps' in dataset else int(
                parts[2])
            o_path = [('/'.join(p.split(':')[0].split('/')) +
                       ':' + p.split(':')[1]) for p in paths]
            line = parts[0] + '\t' + parts[1] + '\t' + parts[2] + \
                '\t' + parts[3] + '\t' + ' '.join(o_path) + '\n'
            filtered_data.append(line)
            cascade_valid_total += 1

    if not unlabel:
        shuffled_time = list(cascades_time_dict.keys())
        random.seed(seed)
        random.shuffle(shuffled_time)
        for count, key in enumerate(shuffled_time):
            if count < cascade_valid_total * train_ratio:
                cascades_type[key] = 0  # training set
            # validation set
            elif count < cascade_valid_total * (1 + train_ratio) / 2:
                cascades_type[key] = 1
            else:
                cascades_type[key] = 2  # test set
    process_and_write_data(filtered_data, cascades_type, observation_time,
                           prediction_time, file_full, file_train, file_val, file_test, file_unlabel, unlabel)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cascades from data.")
    parser.add_argument('--root', type=str,
                        default='Input', help='Root path.')
    parser.add_argument('--dataset', type=str,
                        default='twitter', help='Dataset name.')
    parser.add_argument('--train_ratio', type=float,
                        default=.7, help='The proportion of data to be used for training.')
    parser.add_argument('--unlabel', type=bool, default=False,
                        help='Generate unlabeled data.')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Seed for random number generation.')
    args = parser.parse_args()

    observation_time = config[args.dataset]["Observation Time"]
    prediction_time = config[args.dataset]["Prediction Time"]
    filename = osp.join(args.root, args.dataset, 'raw', 'dataset.txt')
    file_full = osp.join(args.root, args.dataset, 'raw', 'full.txt')
    file_train = osp.join(args.root, args.dataset, 'raw', 'train.txt')
    file_val = osp.join(args.root, args.dataset, 'raw', 'val.txt')
    file_test = osp.join(args.root, args.dataset, 'raw', 'test.txt')
    file_unlabel = osp.join(args.root, args.dataset, 'raw', 'unlabel.txt')
    generate_cascades(
        args.dataset,
        observation_time,
        prediction_time,
        args.unlabel,
        filename,
        file_full,
        file_train,
        file_val,
        file_test,
        file_unlabel,
        args.train_ratio,
        args.seed
    )


if __name__ == "__main__":
    main()
