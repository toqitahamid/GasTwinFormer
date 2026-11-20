"""Unpack the parquet files of this dataset into PNG folders and split lists.

Usage:  python tools/unpack_dataset.py <release_dir> <out_dir>

Needs only the standard library and pyarrow. It writes
  <out_dir>/images/<video>/<video>_frame_<NNNNN>.png        (original image bytes)
  <out_dir>/masks/<video>/<video>_frame_<NNNNN>.png         (original binary mask bytes)
  <out_dir>/masks_class/<video>/<video>_frame_<NNNNN>.png   (original class mask bytes)
  <out_dir>/metadata.csv                                    (diet and animal of every video)
  <out_dir>/train.txt, val.txt, test.txt                    (official split)
  <out_dir>/animal_train.txt, animal_val.txt, animal_test.txt  (by-animal split)
Each list line is <video>/<video>_frame_<NNNNN>. This is the path below images/
and masks/ without the suffix, so it works as an MMSegmentation ann_file with
img_path='images', seg_map_path='masks' and suffix '.png'.
"""
import os, sys, glob, shutil
import pyarrow.parquet as pq

KINDS = (('image', 'images'), ('mask', 'masks'), ('mask_class', 'masks_class'))
OFFICIAL = {'train': 'train.txt', 'validation': 'val.txt', 'test': 'test.txt'}
ANIMAL = {'train': 'animal_train.txt', 'validation': 'animal_val.txt', 'test': 'animal_test.txt'}


def unpack(release, out):
    files = sorted(glob.glob(os.path.join(release, 'data', '*.parquet')))
    off = {p: [] for p in OFFICIAL}
    ani = {p: [] for p in ANIMAL}
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            for r in batch.to_pylist():
                v, fi = r['video_id'], r['frame_index']
                stem = '%s_frame_%05d' % (v, fi)
                for col, folder in KINDS:
                    d = os.path.join(out, folder, v)
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, stem + '.png'), 'wb') as fh:
                        fh.write(r[col]['bytes'])
                off[r['official_split']].append('%s/%s' % (v, stem))
                ani[r['animal_split']].append('%s/%s' % (v, stem))
    for lists, names in ((off, OFFICIAL), (ani, ANIMAL)):
        for part, name in names.items():
            with open(os.path.join(out, name), 'w') as fh:
                fh.write(''.join(s + '\n' for s in sorted(lists[part])))
    shutil.copy(os.path.join(release, 'metadata.csv'), os.path.join(out, 'metadata.csv'))
    print(len(files), 'files; official', {p: len(l) for p, l in off.items()},
          'animal', {p: len(l) for p, l in ani.items()})


if __name__ == '__main__':
    unpack(sys.argv[1], sys.argv[2])
