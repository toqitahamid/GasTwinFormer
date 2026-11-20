# Beef Cattle Methane Emission dataset, written by tools/unpack_dataset.py.
# Change the three ann_file names to animal_train.txt, animal_val.txt and
# animal_test.txt to train on the by-animal split.
dataset_type = 'DietPlumeDataset'
data_root = 'data/'
img_scale = (640, 480)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadDietLabel'),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackMultitaskInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultitaskResize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='LoadDietLabel'),
    dict(type='PackMultitaskInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.txt',
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        pipeline=test_pipeline))

val_evaluator = dict(type='MultitaskEvaluator')
test_evaluator = dict(type='MultitaskEvaluator')
