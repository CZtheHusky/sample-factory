import os
import string
import itertools

replace_target = {
    'explore_goal_locations': "return factory.createLevelApi{\n    episodeLengthSeconds = 120,\n    mazeHeight = 17,\n    mazeWidth = 17,\n    objectCount = 5,\n    roomCount = 9,\n    roomMaxSize = 3,\n}\n",
    'explore_object_locations': "return factory.createLevelApi{\n    episodeLengthSeconds = 120,\n    mazeHeight = 17,\n    mazeWidth = 17,\n    roomCount = 9,\n    roomMaxSize = 3,\n}\n",
    'explore_object_rewards': "return factory.createLevelApi{\n    categoryCount = 8,\n    episodeLengthSeconds = 120,\n    minBadCategory = 4,\n    minGoodCategory = 4,\n    objectCount = 16,\n}\n\n",
    'explore_obstructed_goals': "return factory.createLevelApi{\n    doorsClosed = 0.5,\n    episodeLengthSeconds = 120,\n    mazeHeight = 17,\n    mazeWidth = 17,\n    objectCount = 5,\n    roomCount = 9,\n    roomMaxSize = 3,\n}\n\n",
    'lasertag': "return factory.createLevelApi{\n    episodeLengthSeconds = 240,\n    botCount = 1,\n    color = true,\n    mazeGenerationParams = {\n        height = 21,\n        width = 21,\n        maxRooms = 4,\n        roomMinSize = 5,\n        roomMaxSize = 7,\n        roomSpawnCount = 3,\n    },\n    pickupParams = {\n        pickupCount = 4,\n        weaponCount = 2,\n    },\n}\n"
}


def replace_str(level, combine: list, new=False):
    if level == 'explore_goal_locations':
        assert len(combine) == 4
        return "return factory.createLevelApi{{\n    episodeLengthSeconds = 120,\n    mazeHeight = {0},\n    mazeWidth = {0},\n    objectCount = {1},\n    roomCount = {2},\n    roomMaxSize = {3},\n}}\n".format(
            *combine)
    if level == 'explore_object_locations':
        assert len(combine) == 3
        return "return factory.createLevelApi{{\n    episodeLengthSeconds = 120,\n    mazeHeight = {0},\n    mazeWidth = {0},\n    roomCount = {1},\n    roomMaxSize = {2},\n}}\n".format(
            *combine)
    if level == 'explore_object_rewards':
        if new:
            assert len(combine) == 3
            bad_num = round((1 - combine[1]) * combine[0], 0)
            bad_num = int(bad_num)
            good_num = combine[0] - bad_num
            combine = [combine[0], bad_num, good_num, combine[2]]
        else:
            assert len(combine) == 4
        return "return factory.createLevelApi{{\n    categoryCount = {0},\n    episodeLengthSeconds = 120,\n    minBadCategory = {1},\n    minGoodCategory = {2},\n    objectCount = {3},\n}}\n\n".format(
            *combine)
    if level == 'explore_obstructed_goals':
        assert len(combine) == 5
        return "return factory.createLevelApi{{\n    doorsClosed = {0},\n    episodeLengthSeconds = 120,\n    mazeHeight = {1},\n    mazeWidth = {1},\n    objectCount = {2},\n    roomCount = {3},\n    roomMaxSize = {4},\n}}\n\n".format(
            *combine)
    if level == 'lasertag':
        assert len(combine) == 6
        return "return factory.createLevelApi{{\n    episodeLengthSeconds = 240,\n    botCount = {0},\n    color = true,\n    mazeGenerationParams = {{\n        height = {1},\n        width = {1},\n        maxRooms = {2},\n        roomMinSize = {3},\n        roomMaxSize = {4},\n        roomSpawnCount = {5},\n    }},\n    pickupParams = {{\n        pickupCount = 4,\n        weaponCount = 2,\n    }},\n}}\n".format(*combine)


combinations = {
    'explore_goal_locations': [[11, 17, 23], [3, 5, 7, 9], [7, 9, 11], [3, 5]],
    'explore_object_locations': [[11, 17, 23, 29], [5, 7, 9, 11], [3, 5, 7]],
    'explore_object_rewards': [[10, 12, 14], [3, 4, 5], [3, 4, 5], [20, 24, 28]],
    'explore_obstructed_goals': [[0.25, 0.5, 0.75], [11, 17, 23], [3, 5, 7], [7, 11], [3, 5]],
    'lasertag': [[1, 3, 5], [15, 21, 27], [2, 4, 8], [3], [7, 9], [3, 7]],
}

combinations1 = {
    'explore_goal_locations': [[11, 22, 33], [3, 9, 15, 21], [7, 14, 21], [3, 9]],
    'explore_object_locations': [[11, 22, 33, 44], [5, 11, 17, 23], [3, 9, 15]],
    'explore_object_rewards': [[10, 20, 30], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [20, 40, 60]],
    'explore_obstructed_goals': [[0.25, 0.5, 0.75], [11, 22, 33], [10, 20, 30], [11, 22], [3, 9]],
    'lasertag': [[1, 5, 10], [15, 30, 45], [5, 10, 15], [3], [5, 11], [5, 10]],
}

base_path = './base_levels'
base_path = os.path.abspath(base_path)
generate_num = 50
father_path = os.path.dirname(base_path)
bases = os.listdir(base_path)
for base_file in bases:
    file_name, suffix = base_file.split('.')
    target_path = os.path.join(father_path, 'old_gen_levels', file_name)
    n_target_path = os.path.join(father_path, 'new_gen_levels', file_name + '_new')
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(n_target_path, exist_ok=True)
    with open(os.path.join(base_path, base_file), 'r') as f:
        context = f.read()
    combines = itertools.product(*combinations[file_name], repeat=1)
    combines = list(combines)
    for idx, combine in enumerate(combines):
        with open(os.path.join(target_path, file_name + '-' + str(idx) + '.lua'), 'w') as f:
            f.write(context.replace(replace_target[file_name], replace_str(file_name, combine)))
    combines = itertools.product(*combinations1[file_name], repeat=1)
    combines = list(combines)
    for idx, combine in enumerate(combines):
        with open(os.path.join(n_target_path, file_name + '-' + str(idx) + '.lua'), 'w') as f:
            f.write(context.replace(replace_target[file_name], replace_str(file_name, combine, True)))
    list2print = []
    for files in os.listdir(target_path):
        f_name, _ = files.split('.')
        list2print.append(f_name)
    print(file_name, 'num: ', len(list2print))
    print(list2print)
    list2print = []
    for files in os.listdir(n_target_path):
        f_name, _ = files.split('.')
        list2print.append(f_name)
    print(file_name + '_new', 'num: ', len(list2print))
    print(list2print)

# rm -rf ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels/old_gen_levels && rm -rf ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels/new_gen_levels && cp -r ./level_gen/old_gen_levels ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels && cp -r ./level_gen/new_gen_levels ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels