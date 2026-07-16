# TODO: Write wrapper for torchvision.datasets.Places365 that also has support set

# Link: http://places2.csail.mit.edu/download-private.html
#
# For example: Small images (256 * 256):
#  + Train images. 24GB. MD5: 53ca1c756c3d1e7809517cc47c5561c5
#  + Validation images. 501M. MD5: e27b17d8d44f4af9a78502beb927f808
#  + Test images. 4.4G. MD5: f532f6ad7b582262a2ec8009075e186b --> Labels are not provided, should be -1 as default
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive
from torchvision.datasets import Places365

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)
    
    BASE_DIR = './datasets/Places365'
    FILE = 'test_256.tar'
    md5 = 'f532f6ad7b582262a2ec8009075e186b'

    integrity = check_integrity(os.path.join(BASE_DIR, FILE), md5)
    print('Integrity check:', integrity)

    if not os.path.isdir(f'{BASE_DIR}/test_256'):
        print('Archive not extracted. Extracting...')
        extract_archive(os.path.join(BASE_DIR, FILE), BASE_DIR)
    else:
        print('Archive already extracted.')