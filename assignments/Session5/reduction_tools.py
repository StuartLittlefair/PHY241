from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from collections import OrderedDict
import os
import warnings
import six

import ccdproc

import numpy as np


DEFAULT_IMAGE_UNIT = "adu"

# The dictionary below is used to map the dtype of the image being
# reduced to the dtype of the output. The assumption is that the output
# is typically some kind of floating point, but that there is no need
# for very high precision output given relatively low resolution
# input.
REDUCE_IMAGE_DTYPE_MAPPING = {
    'uint8': 'float32',
    'int8': 'float32',
    'uint16': 'float32',
    'int16': 'float32',
    'float32': 'float32',
    'uint32': 'float64',
    'int32': 'float64',
    'float64': 'float64'
}

# The limit below is used  by the combining function to decide whether or
# not the image should be broken up into chunks.
DEFAULT_MEMORY_LIMIT = 4e9  # roughly 4GB


class ReducerBase:
    def __init__(self, *arg, **kwd):
        self._apply_to = kwd.pop('apply_to', None)
        self._destination = kwd.pop('destination', None)

    @property
    def destination(self):
        return self._destination

    @property
    def apply_to(self):
        return self._apply_to


class Reduction(ReducerBase):
    """
    Primary widget for performing a logical reduction step (e.g. dark
    subtraction or flat correction).
    """
    def __init__(self, *arg, **kwd):
        allow_flat = kwd.pop('flat_correct', True)
        allow_dark = kwd.pop('dark_subtract', True)
        allow_dark_scale = kwd.pop('dark_scale', True)
        allow_bias = kwd.pop('bias_subtract', True)
        allow_copy = kwd.pop('copy_only', False)
        self.image_collection = kwd.pop('input_image_collection', None)
        self._master_source = kwd.pop('master_source', None)
        super(Reduction, self).__init__(*arg, **kwd)

        self._bias_calib = BiasSubtract(master_source=self._master_source)
        self._dark_calib = DarkSubtract(master_source=self._master_source, dark_scale=allow_dark_scale)
        self._flat_calib = FlatCorrect(master_source=self._master_source)

        self.children = []

        if allow_copy:
            self._copy_only = CopyFiles()
            self.add_child(self._copy_only)
        else:
            if allow_bias:
                self.add_child(self._bias_calib)
            if allow_dark:
                self.add_child(self._dark_calib)
            if allow_flat:
                self.add_child(self._flat_calib)

    def add_child(self, child):
        self.children.append(child)

    def action(self):
        if not self.image_collection:
            raise ValueError("No images to reduce")

        # Refresh in case files have been added since the widget was created.
        self.image_collection.refresh()

        # Only refresh the master_source if it exists. No need to error check
        # the main image_collection because a sensible error is raised if it
        # does not exist.
        if self._master_source:
            self._master_source.refresh()

        # Suppress warnings that come up here...mostly about HIERARCH keywords
        warnings.filterwarnings('ignore')
        try:
            current_file = 0
            for hdu, fname in self.image_collection.hdus(return_fname=True,
                                                         save_location=self.destination,
                                                         overwrite=True,
                                                         **self.apply_to):
                current_file += 1
                try:
                    unit = hdu.header['BUNIT']
                except KeyError:
                    unit = DEFAULT_IMAGE_UNIT
                ccd = ccdproc.CCDData(hdu.data, meta=hdu.header, unit=unit)
                for child in self.children:
                    ccd = child.action(ccd)

                input_dtype = hdu.data.dtype.name
                hdu_tmp = ccd.to_hdu()[0]
                hdu.header = hdu_tmp.header
                hdu.data = hdu_tmp.data
                desired_dtype = REDUCE_IMAGE_DTYPE_MAPPING[str(input_dtype)]
                if desired_dtype != hdu.data.dtype:
                    hdu.data = hdu.data.astype(desired_dtype)

                # Workaround to ensure uint16 images are handled properly.
                if 'bzero' in hdu.header:
                    # Check for the unsigned int16 case, and if our data type
                    # is no longer uint16, delete BZERO and BSCALE
                    header_unsigned_int = ((hdu.header['bscale'] == 1) and
                                           (hdu.header['bzero'] == 32768))
                    if (header_unsigned_int and (hdu.data.dtype != np.dtype('uint16'))):
                        del hdu.header['bzero'], hdu.header['bscale']
        except IOError:
            print("One or more of the reduced images already exists. Delete "
                  "those files and try again. This class will NOT "
                  "overwrite existing files.")


class CopyFiles:
    def action(self, ccd):
        return ccd


class GroupBy:
    def __init__(self, *args, **kwd):
        self._image_source = kwd.pop('image_source', None)
        input_value = kwd.pop('value', '')
        self.keyword_list = [k.strip() for k in input_value.split(',')]

    def groups(self, apply_to):
        if self.keyword_list == ['']:
            # Return an empty dictionary by default if there is no grouping
            return [{}]

        self._image_source.refresh()

        from copy import deepcopy
        tmp_coll = deepcopy(self._image_source)
        tmp_coll._find_keywords_by_values(**apply_to)
        mask = tmp_coll.summary['file'].mask
        # Note the logical not below; mask indicates which values
        # should be EXCLUDED.
        filtered_table = tmp_coll.summary[~mask]
        grouped_table = filtered_table.group_by(self.keyword_list)
        combine_groups = grouped_table.groups.keys
        group_list = []
        for row in combine_groups:
            d = {c: row[c] for c in combine_groups.colnames}
            group_list.append(d)
        return group_list


class Combiner(ReducerBase):
    """
    Class for combining frames to make master darks, biasses, flats
    """
    def __init__(self, *args, **kwd):
        group_by_in = kwd.pop('group_by', '')
        self._image_source = kwd.pop('image_source', None)
        self._file_base_name = kwd.pop('file_name_base', 'master')
        super(Combiner, self).__init__(*args, **kwd)
        self._combine_method = kwd.pop('combine_method', 'average')
        self._scaling = kwd.pop('scaling', None)
        if self._scaling == 'mean':
            self._scaling_func = lambda arr: 1/np.ma.average(arr)
        elif self._scaling == 'median':
            self._scaling_func = lambda arr: 1/np.ma.median(arr)

        self._combined = None
        self._group_by = GroupBy(value=group_by_in, image_source=self._image_source)

    @property
    def combined(self):
        """
        The combined image.
        """
        return self._combined

    @property
    def image_source(self):
        return self._image_source

    def action(self):
        """
        Combine files by groups
        """
        # Refresh image collection in case files were added after widget was
        # created.
        self.image_source.refresh()

        # Suppress warnings that come up here...mostly about HIERARCH keywords
        warnings.filterwarnings('ignore')
        groups_to_combine = self._group_by.groups(self.apply_to)
        for idx, combo_group in enumerate(groups_to_combine):
            combined = self._action_for_one_group(combo_group)
            name_addons = ['_'.join([str(k), str(v)])
                           for k, v in six.iteritems(combo_group)]
            fname = [self._file_base_name]
            fname.extend(name_addons)
            fname = '_'.join(fname) + '.fit'
            dest_path = os.path.join(self.destination, fname)
            combined.write(dest_path, overwrite=True)
            self._combined = combined

    def _action_for_one_group(self, filter_dict=None):
        combined_dict = self.apply_to.copy()
        if filter_dict is not None:
            combined_dict.update(filter_dict)

        file_list = [os.path.join(self.image_source.location, f) for f in
                     self.image_source.files_filtered(**combined_dict)]

        combine_keyword_args = {}
        combine_keyword_args['method'] = self._combine_method

        if self._scaling:
            combine_keyword_args['scale'] = self._scaling_func

        combined = ccdproc.combine(file_list,
                                   mem_limit=DEFAULT_MEMORY_LIMIT,
                                   **combine_keyword_args)

        sample_image = ccdproc.CCDData.read(file_list[0])
        combined.header = sample_image.header
        combined.header['master'] = True
        if combined.data.dtype != sample_image.dtype:
            combined.data = np.array(combined.data, dtype=sample_image.dtype)
        try:
            if isinstance(combined.uncertainty.array, np.ma.masked_array):
                combined.uncertainty.array = np.array(combined.uncertainty.array)
        except AttributeError:
            pass

        # Do not keep the mask or uncertainty if the data has neither
        if sample_image.mask is None and sample_image.uncertainty is None:
            combined.mask = None
            combined.uncertainty = None
        return combined


class CalibrationStep:
    """
    Represents a calibration step that corresponds to a ccdproc command
    """
    def __init__(self, *args, **kwd):
        self._master_source = kwd.pop('master_source', None)
        self._image_cache = {}
        self._match_on = []

    @property
    def match_on(self):
        """
        List of keywords whose values should match in the image being
        calibated and the calibration image.
        """
        return self._match_on

    @match_on.setter
    def match_on(self, value):
        self._match_on = value

    def _master_image(self, selector, closest=None):
        """
        Identify appropriate master and return as `ccdproc.CCDData`.
        Parameters
        ----------
        selector : dict-like
            Dictionary of key/value pairs that uniquely select the appropriate
            master image.
        closest : str, optional
            Name of keyword from ``selector`` whose value needs only be
            closest to the value in the dictionary instead of being an
            exact match.
        """
        if not self._master_source:
            raise RuntimeError("No source provided for master.")
        file_name = self._master_source.files_filtered(master=True,
                                                       **selector)
        if len(file_name) > 1:
            raise RuntimeError("Well, crap. Should only be one master but "
                               "found these matches: "
                               "{} for {}.".format(file_name, selector))
        elif len(file_name) == 0:
            if closest is None:
                raise RuntimeError("No master found for {}".format(selector))
            else:
                new_select = selector.copy()
                del new_select[closest]
                file_name = self._master_source.files_filtered(master=True,
                                                               **new_select)
                master_table = self._master_source.summary
                min_dist = 1e20
                for name in file_name:
                    match = master_table['file'] == name
                    distance = abs(master_table[closest][match] -
                                   selector[closest])
                    if distance <= min_dist:
                        best_match = name
                        min_dist = distance
                file_name = [best_match]
        file_name = file_name[0]
        path = os.path.join(self._master_source.location, file_name)
        try:
            return self._image_cache[path]
        except KeyError:
            # Try getting the unit form the FITS file, but force it to ADU
            try:
                self._image_cache[path] = ccdproc.CCDData.read(path)
            except ValueError:
                self._image_cache[path] = \
                    ccdproc.CCDData.read(path, unit=DEFAULT_IMAGE_UNIT)
            return self._image_cache[path]


class BiasSubtract(CalibrationStep):
    def __init__(self, bias_image=None, **kwd):
        super(BiasSubtract, self).__init__(**kwd)

    def action(self, ccd):
        select_dict = {'imagetyp': 'bias'}
        try:
            master = self._master_image(select_dict)
        except:
            select_dict = {'imagetyp': 'bias frame'}
            master = self._master_image(select_dict)
        return ccdproc.subtract_bias(ccd, master)


class DarkSubtract(CalibrationStep):
    def __init__(self, bias_image=None, **kwd):
        super(DarkSubtract, self).__init__(**kwd)
        self.scale = kwd.pop('dark_scale', False)
        self.match_on = ['exposure']

    def action(self, ccd):
        from astropy import units as u
        select_dict = {'imagetyp': 'dark frame'}
        for keyword in self.match_on:
            if keyword in select_dict:
                raise ValueError("Keyword {} already has a value set".format(keyword))
            select_dict[keyword] = ccd.header[keyword]
        if self.scale:
            try:
                master = self._master_image(select_dict, closest=self.match_on[0])
            except:
                select_dict['imagetyp'] = 'dark'
                master = self._master_image(select_dict, closest=self.match_on[0])
            if 'subbias' not in master.meta:
                raise RuntimeError("Bias has not been subtracted from dark, "
                                   "so cannot scale dark")
        else:
            try:
                master = self._master_image(select_dict)
            except:
                select_dict['imagetyp'] = 'dark'
                master = self._master_image(select_dict)

        return ccdproc.subtract_dark(ccd, master,
                                        exposure_time='exposure',
                                        exposure_unit=u.second,
                                        scale=self.scale)


class FlatCorrect(CalibrationStep):
    def __init__(self, bias_image=None, **kwd):
        super(FlatCorrect, self).__init__(**kwd)
        self.match_on = ['filter']

    def action(self, ccd):
        select_dict = {'imagetyp': 'flat field'}
        for keyword in self.match_on:
            if keyword in select_dict:
                raise ValueError("Keyword {} already has a value set".format(keyword))
            select_dict[keyword] = ccd.header[keyword]
        try:
            master = self._master_image(select_dict)
        except:
            select_dict['imagetyp'] = 'flat'
            master = self._master_image(select_dict)

        return ccdproc.flat_correct(ccd, master)
