import csv
import os.path as osp

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

# Index of each diet in the classification head.
DIET_CLASSES = ('mixed_diet', 'high_forage', 'high_grain')


@DATASETS.register_module()
class DietPlumeDataset(BaseSegDataset):
    """Beef Cattle Methane Emission dataset.

    Each frame has a binary plume mask (0 background, 1 plume) and the diet of
    the animal. The diet label comes from ``metadata.csv`` in ``data_root``,
    which lists the diet of every video. The video of a frame is the part of the
    file name before ``_frame``.

    ``tools/unpack_dataset.py`` writes the folders and split lists that this
    class reads.
    """
    METAINFO = dict(classes=('background', 'plume'),
                    palette=[[0, 0, 0], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 metadata_file='metadata.csv',
                 **kwargs) -> None:
        self.metadata_file = metadata_file
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=reduce_zero_label,
                         **kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        diet_of_video = {}
        with open(osp.join(self.data_root, self.metadata_file)) as f:
            for row in csv.DictReader(f):
                diet_of_video[row['video_id']] = DIET_CLASSES.index(row['diet'])
        for item in data_list:
            name = osp.basename(item['img_path'])
            video = name.split('_frame')[0]
            item['diet_label'] = diet_of_video[video]
            item['diet_type'] = DIET_CLASSES[item['diet_label']]
        return data_list
