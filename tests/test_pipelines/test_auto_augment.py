import copy

import mmcv
import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmcls.datasets.builder import PIPELINES


def construct_toy_data():
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']
    return results


def construct_toy_data_photometric():
    img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']
    return results


def test_shear():
    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid pad_val
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, pad_val=(0, 0))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, direction='diagonal')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude = 0, therefore no shear
    results = construct_toy_data()
    transform = dict(type='Shear', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob = 0, therefore no shear
    results = construct_toy_data()
    transform = dict(type='Shear', magnitude=0.5, prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test shear horizontally, magnitude=1
    results = construct_toy_data()
    transform = dict(
        type='Shear', magnitude=1, pad_val=0, prob=1., random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    sheared_img = np.array([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 9, 10]],
                           dtype=np.uint8)
    sheared_img = np.stack([sheared_img, sheared_img, sheared_img], axis=-1)
    assert (results['img'] == sheared_img).all()
    assert (results['img'] == results['img2']).all()

    # test shear vertically, magnitude=-1
    results = construct_toy_data()
    transform = dict(
        type='Shear',
        magnitude=-1,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    sheared_img = np.array([[1, 6, 11, 0], [5, 10, 0, 0], [9, 0, 0, 0]],
                           dtype=np.uint8)
    sheared_img = np.stack([sheared_img, sheared_img, sheared_img], axis=-1)
    assert (results['img'] == sheared_img).all()

    # test shear vertically, magnitude=1, random_negative_prob=1
    results = construct_toy_data()
    transform = dict(
        type='Shear',
        magnitude=1,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    sheared_img = np.array([[1, 6, 11, 0], [5, 10, 0, 0], [9, 0, 0, 0]],
                           dtype=np.uint8)
    sheared_img = np.stack([sheared_img, sheared_img, sheared_img], axis=-1)
    assert (results['img'] == sheared_img).all()

    # test auto aug with shear
    results = construct_toy_data()
    policies = [[transform]]
    autoaug = dict(type='AutoAugment', policies=policies)
    pipeline = build_from_cfg(autoaug, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == sheared_img).all()


def test_translate():
    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid pad_val
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=0.5, pad_val=(0, 0))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=0.5, direction='diagonal')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude=0, therefore no translate
    results = construct_toy_data()
    transform = dict(type='Translate', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=0, therefore no translate
    results = construct_toy_data()
    transform = dict(type='Translate', magnitude=1., prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test translate horizontally, magnitude=0.5
    results = construct_toy_data()
    transform = dict(
        type='Translate',
        magnitude=0.5,
        pad_val=0,
        prob=1.,
        random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    translated_img = np.array([[0, 0, 1, 2], [0, 0, 5, 6], [0, 0, 9, 10]],
                              dtype=np.uint8)
    translated_img = np.stack([translated_img, translated_img, translated_img],
                              axis=-1)
    assert (results['img'] == translated_img).all()
    assert (results['img'] == results['img2']).all()

    # test translate vertically, magnitude=-0.5
    results = construct_toy_data()
    transform = dict(
        type='Translate',
        magnitude=-0.5,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    translated_img = np.array([[9, 10, 11, 12], [0, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=np.uint8)
    translated_img = np.stack([translated_img, translated_img, translated_img],
                              axis=-1)
    assert (results['img'] == translated_img).all()

    # test translate vertically, magnitude=0.5, random_negative_prob=1
    results = construct_toy_data()
    transform = dict(
        type='Translate',
        magnitude=0.5,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    translated_img = np.array([[9, 10, 11, 12], [0, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=np.uint8)
    translated_img = np.stack([translated_img, translated_img, translated_img],
                              axis=-1)
    assert (results['img'] == translated_img).all()


def test_rotate():
    # test assertion for invalid type of angle
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', angle=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of center
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', angle=90., center=0)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid lenth of center
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', angle=90., center=(0, ))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid scale
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', angle=90., scale=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid pad_val
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', angle=90., pad_val=(0, 0))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', angle=90., prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', angle=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when angle=0, therefore no rotation
    results = construct_toy_data()
    transform = dict(type='Rotate', angle=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when angle=360, therefore no rotation
    results = construct_toy_data()
    transform = dict(type='Rotate', angle=360., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=0, therefore no rotation
    results = construct_toy_data()
    transform = dict(type='Rotate', angle=90., prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test rotate clockwise, angle=30.
    results = construct_toy_data()
    transform = dict(
        type='Rotate', angle=30., pad_val=0, prob=1., random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    rotated_img = np.array([[5, 2, 2, 0], [9, 6, 7, 4], [0, 11, 11, 8]],
                           dtype=np.uint8)
    rotated_img = np.stack([rotated_img, rotated_img, rotated_img], axis=-1)
    assert (results['img'] == rotated_img).all()
    assert (results['img'] == results['img2']).all()

    # test rotate clockwise, angle=90, center=(1,1)
    results = construct_toy_data()
    transform = dict(
        type='Rotate',
        angle=90.,
        center=(1, 1),
        prob=1.,
        random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    rotated_img = np.array([[9, 5, 1, 128], [10, 6, 2, 128], [11, 7, 3, 128]],
                           dtype=np.uint8)
    rotated_img = np.stack([rotated_img, rotated_img, rotated_img], axis=-1)
    assert (results['img'] == rotated_img).all()
    assert (results['img'] == results['img2']).all()

    # test rotate counter-clockwise, angle=90.
    results = construct_toy_data()
    transform = dict(
        type='Rotate', angle=-90., pad_val=0, prob=1., random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    rotated_img = np.array([[4, 8, 12, 0], [3, 7, 11, 0], [2, 6, 10, 0]],
                           dtype=np.uint8)
    rotated_img = np.stack([rotated_img, rotated_img, rotated_img], axis=-1)
    assert (results['img'] == rotated_img).all()
    assert (results['img'] == results['img2']).all()

    # test rotate counter-clockwise, angle=90, random_negative_prob=1
    results = construct_toy_data()
    transform = dict(
        type='Rotate', angle=-90., pad_val=0, prob=1., random_negative_prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    rotated_img = np.array([[0, 10, 6, 2], [0, 11, 7, 3], [0, 12, 8, 4]],
                           dtype=np.uint8)
    rotated_img = np.stack([rotated_img, rotated_img, rotated_img], axis=-1)
    assert (results['img'] == rotated_img).all()
    assert (results['img'] == results['img2']).all()


def test_auto_contrast():
    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='AutoContrast', prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when prob=0, therefore no auto_contrast
    results = construct_toy_data()
    transform = dict(type='AutoContrast', prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=1
    results = construct_toy_data()
    transform = dict(type='AutoContrast', prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    auto_contrasted_img = np.array(
        [[0, 23, 46, 69], [92, 115, 139, 162], [185, 208, 231, 255]],
        dtype=np.uint8)
    auto_contrasted_img = np.stack(
        [auto_contrasted_img, auto_contrasted_img, auto_contrasted_img],
        axis=-1)
    assert (results['img'] == auto_contrasted_img).all()
    assert (results['img'] == results['img2']).all()


def test_invert():
    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Invert', prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when prob=0, therefore no invert
    results = construct_toy_data()
    transform = dict(type='Invert', prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=1
    results = construct_toy_data()
    transform = dict(type='Invert', prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    inverted_img = np.array(
        [[254, 253, 252, 251], [250, 249, 248, 247], [246, 245, 244, 243]],
        dtype=np.uint8)
    inverted_img = np.stack([inverted_img, inverted_img, inverted_img],
                            axis=-1)
    assert (results['img'] == inverted_img).all()
    assert (results['img'] == results['img2']).all()


def test_equalize(nb_rand_test=100):

    def _imequalize(img):
        # equalize the image using PIL.ImageOps.equalize
        from PIL import ImageOps, Image
        img = Image.fromarray(img)
        equalized_img = np.asarray(ImageOps.equalize(img))
        return equalized_img

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Equalize', prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when prob=0, therefore no equalize
    results = construct_toy_data()
    transform = dict(type='Equalize', prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=1 with randomly sampled image.
    results = construct_toy_data()
    transform = dict(type='Equalize', prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    for _ in range(nb_rand_test):
        img = np.clip(np.random.normal(0, 1, (1000, 1200, 3)) * 260, 0,
                      255).astype(np.uint8)
        results['img'] = img
        results = pipeline(copy.deepcopy(results))
        assert (results['img'] == _imequalize(img)).all()


def test_solarize():
    # test assertion for invalid type of thr
    with pytest.raises(AssertionError):
        transform = dict(type='Solarize', thr=(1, 2))
        build_from_cfg(transform, PIPELINES)

    # test case when prob=0, therefore no solarize
    results = construct_toy_data_photometric()
    transform = dict(type='Solarize', thr=128, prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when thr=256, therefore no solarize
    results = construct_toy_data_photometric()
    transform = dict(type='Solarize', thr=256, prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when thr=128
    results = construct_toy_data_photometric()
    transform = dict(type='Solarize', thr=128, prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    img_solarized = np.array([[0, 127, 0], [1, 127, 1], [2, 126, 2]],
                             dtype=np.uint8)
    img_solarized = np.stack([img_solarized, img_solarized, img_solarized],
                             axis=-1)
    assert (results['img'] == img_solarized).all()
    assert (results['img'] == results['img2']).all()

    # test case when thr=100
    results = construct_toy_data_photometric()
    transform = dict(type='Solarize', thr=100, prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    img_solarized = np.array([[0, 127, 0], [1, 128, 1], [2, 126, 2]],
                             dtype=np.uint8)
    img_solarized = np.stack([img_solarized, img_solarized, img_solarized],
                             axis=-1)
    assert (results['img'] == img_solarized).all()
    assert (results['img'] == results['img2']).all()


def test_posterize():
    # test assertion for invalid type of bits
    with pytest.raises(AssertionError):
        transform = dict(type='Posterize', bits=4.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of bits
    with pytest.raises(AssertionError):
        transform = dict(type='Posterize', bits=10)
        build_from_cfg(transform, PIPELINES)

    # test case when prob=0, therefore no posterize
    results = construct_toy_data_photometric()
    transform = dict(type='Posterize', bits=4, prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when bits=8, therefore no solarize
    results = construct_toy_data_photometric()
    transform = dict(type='Posterize', bits=8, prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when bits=1
    results = construct_toy_data_photometric()
    transform = dict(type='Posterize', bits=1, prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    img_posterized = np.array([[0, 128, 128], [0, 0, 128], [0, 128, 128]],
                              dtype=np.uint8)
    img_posterized = np.stack([img_posterized, img_posterized, img_posterized],
                              axis=-1)
    assert (results['img'] == img_posterized).all()
    assert (results['img'] == results['img2']).all()

    # test case when bits=3
    results = construct_toy_data_photometric()
    transform = dict(type='Posterize', bits=3, prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    img_posterized = np.array([[0, 128, 224], [0, 96, 224], [0, 128, 224]],
                              dtype=np.uint8)
    img_posterized = np.stack([img_posterized, img_posterized, img_posterized],
                              axis=-1)
    assert (results['img'] == img_posterized).all()
    assert (results['img'] == results['img2']).all()


def test_contrast(nb_rand_test=100):

    def _adjust_contrast(img, factor):
        from PIL.ImageEnhance import Contrast
        from PIL import Image
        # Image.fromarray defaultly supports RGB, not BGR.
        # convert from BGR to RGB
        img = Image.fromarray(img[..., ::-1], mode='RGB')
        contrasted_img = Contrast(img).enhance(factor)
        # convert from RGB to BGR
        return np.asarray(contrasted_img)[..., ::-1]

    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Contrast', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Contrast', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(
            type='Contrast', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude=0, therefore no adjusting contrast
    results = construct_toy_data_photometric()
    transform = dict(type='Contrast', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=0, therefore no adjusting contrast
    results = construct_toy_data_photometric()
    transform = dict(type='Contrast', magnitude=1., prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=1 with randomly sampled image.
    results = construct_toy_data()
    for _ in range(nb_rand_test):
        magnitude = np.random.uniform() * np.random.choice([-1, 1])
        transform = dict(
            type='Contrast',
            magnitude=magnitude,
            prob=1.,
            random_negative_prob=0.)
        pipeline = build_from_cfg(transform, PIPELINES)
        img = np.clip(np.random.uniform(0, 1, (1200, 1000, 3)) * 260, 0,
                      255).astype(np.uint8)
        results['img'] = img
        results = pipeline(copy.deepcopy(results))
        # Note the gap (less_equal 1) between PIL.ImageEnhance.Contrast
        # and mmcv.adjust_contrast comes from the gap that converts from
        # a color image to gray image using mmcv or PIL.
        np.testing.assert_allclose(
            results['img'],
            _adjust_contrast(img, 1 + magnitude),
            rtol=0,
            atol=1)


def test_color_transform():
    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='ColorTransform', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='ColorTransform', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(
            type='ColorTransform', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude=0, therefore no color transform
    results = construct_toy_data_photometric()
    transform = dict(type='ColorTransform', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=0, therefore no color transform
    results = construct_toy_data_photometric()
    transform = dict(type='ColorTransform', magnitude=1., prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when magnitude=-1, therefore got gray img
    results = construct_toy_data_photometric()
    transform = dict(
        type='ColorTransform', magnitude=-1., prob=1., random_negative_prob=0)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    img_gray = mmcv.bgr2gray(results['ori_img'])
    img_gray = np.stack([img_gray, img_gray, img_gray], axis=-1)
    assert (results['img'] == img_gray).all()

    # test case when magnitude=0.5
    results = construct_toy_data_photometric()
    transform = dict(
        type='ColorTransform', magnitude=.5, prob=1., random_negative_prob=0)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    img_r = np.round(
        np.clip((results['ori_img'] * 0.5 + img_gray * 0.5), 0,
                255)).astype(results['ori_img'].dtype)
    assert (results['img'] == img_r).all()
    assert (results['img'] == results['img2']).all()

    # test case when magnitude=0.3, random_negative_prob=1
    results = construct_toy_data_photometric()
    transform = dict(
        type='ColorTransform', magnitude=.3, prob=1., random_negative_prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    img_r = np.round(
        np.clip((results['ori_img'] * 0.7 + img_gray * 0.3), 0,
                255)).astype(results['ori_img'].dtype)
    assert (results['img'] == img_r).all()
    assert (results['img'] == results['img2']).all()


def test_brightness(nb_rand_test=100):

    def _adjust_brightness(img, factor):
        # adjust the brightness of image using
        # PIL.ImageEnhance.Brightness
        from PIL.ImageEnhance import Brightness
        from PIL import Image
        img = Image.fromarray(img)
        brightened_img = Brightness(img).enhance(factor)
        return np.asarray(brightened_img)

    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Brightness', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Brightness', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(
            type='Brightness', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude=0, therefore no adjusting brightness
    results = construct_toy_data_photometric()
    transform = dict(type='Brightness', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=0, therefore no adjusting brightness
    results = construct_toy_data_photometric()
    transform = dict(type='Brightness', magnitude=1., prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=1 with randomly sampled image.
    results = construct_toy_data()
    for _ in range(nb_rand_test):
        magnitude = np.random.uniform() * np.random.choice([-1, 1])
        transform = dict(
            type='Brightness',
            magnitude=magnitude,
            prob=1.,
            random_negative_prob=0.)
        pipeline = build_from_cfg(transform, PIPELINES)
        img = np.clip(np.random.uniform(0, 1, (1200, 1000, 3)) * 260, 0,
                      255).astype(np.uint8)
        results['img'] = img
        results = pipeline(copy.deepcopy(results))
        np.testing.assert_allclose(
            results['img'],
            _adjust_brightness(img, 1 + magnitude),
            rtol=0,
            atol=1)


def test_sharpness(nb_rand_test=100):

    def _adjust_sharpness(img, factor):
        # adjust the sharpness of image using
        # PIL.ImageEnhance.Sharpness
        from PIL.ImageEnhance import Sharpness
        from PIL import Image
        img = Image.fromarray(img)
        sharpened_img = Sharpness(img).enhance(factor)
        return np.asarray(sharpened_img)

    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Sharpness', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Sharpness', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(
            type='Sharpness', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude=0, therefore no adjusting sharpness
    results = construct_toy_data_photometric()
    transform = dict(type='Sharpness', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=0, therefore no adjusting sharpness
    results = construct_toy_data_photometric()
    transform = dict(type='Sharpness', magnitude=1., prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=1 with randomly sampled image.
    results = construct_toy_data()
    for _ in range(nb_rand_test):
        magnitude = np.random.uniform() * np.random.choice([-1, 1])
        transform = dict(
            type='Sharpness',
            magnitude=magnitude,
            prob=1.,
            random_negative_prob=0.)
        pipeline = build_from_cfg(transform, PIPELINES)
        img = np.clip(np.random.uniform(0, 1, (1200, 1000, 3)) * 260, 0,
                      255).astype(np.uint8)
        results['img'] = img
        results = pipeline(copy.deepcopy(results))
        np.testing.assert_allclose(
            results['img'][1:-1, 1:-1],
            _adjust_sharpness(img, 1 + magnitude)[1:-1, 1:-1],
            rtol=0,
            atol=1)
