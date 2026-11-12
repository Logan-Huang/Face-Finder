"""RAW regressions; set FACEFINDER_RAW_SAMPLES to a directory for camera-file checks."""
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image

from facefinder.engine import PhotoError, load_image
from facefinder.web import create_app


class NoFaces:
    def faces(self, img):
        assert img.dtype == np.uint8 and img.shape[2] == 3
        return []


@pytest.mark.parametrize('extension', ['CR2', 'CR3', 'CRW', 'RAF'])
def test_corrupt_raw_is_reported(extension):
    name = 'broken.' + extension
    with pytest.raises(PhotoError):
        load_image(io.BytesIO(b'not raw'), name)
    client = create_app(NoFaces()).test_client()
    scan = client.post('/api/scan', data={
        'file': (io.BytesIO(b'not raw'), name), 'people': '[[[1,0]]]'})
    assert 'error' in scan.get_json()
    assert client.post('/api/thumb', data={
        'file': (io.BytesIO(b'not raw'), name)}).status_code == 422


sample_dir = os.environ.get('FACEFINDER_RAW_SAMPLES')
samples = sorted(p for p in Path(sample_dir).glob('*')
                 if p.suffix.lower() in {'.cr2', '.cr3', '.crw', '.raf'}) if sample_dir else []


@pytest.mark.parametrize('path', samples or [None])
def test_real_raw_end_to_end(path):
    if path is None:
        pytest.skip('Set FACEFINDER_RAW_SAMPLES to test genuine camera files')
    data = path.read_bytes()
    disk = load_image(path)
    assert np.array_equal(disk, load_image(io.BytesIO(data), path.name))
    client = create_app(NoFaces()).test_client()
    ref = client.post('/api/reference', data={'files': (io.BytesIO(data), path.name)}).get_json()
    assert ref['errors'] == [f'{path.name}: no face found']
    scan = client.post('/api/scan', data={
        'file': (io.BytesIO(data), path.name), 'people': json.dumps([[[1, 0]]])}).get_json()
    assert 'error' not in scan
    thumb = client.post('/api/thumb', data={'file': (io.BytesIO(data), path.name)})
    assert thumb.status_code == 200
    image = Image.open(io.BytesIO(thumb.data))
    assert max(image.size) <= 480
    assert abs(image.width / image.height - disk.shape[1] / disk.shape[0]) < 0.02
    exported = client.post('/api/zip', data={'files': (io.BytesIO(data), path.name)})
    with zipfile.ZipFile(io.BytesIO(exported.data)) as archive:
        assert archive.read(path.name) == data


def test_image_stream_with_numeric_name():
    # Temporary uploaded streams may expose a file descriptor as their name.
    class Upload(io.BytesIO):
        name = 7
    stream = Upload()
    Image.new('RGB', (8, 6), 'red').save(stream, 'PNG')
    stream.seek(0)
    assert load_image(stream).shape == (6, 8, 3)
