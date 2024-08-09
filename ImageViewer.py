from collections import namedtuple
from enum import Enum
import math
import cv2 as cv
import numpy as np

image_viewers = {}


def getImageViewer(window_name):
    if not window_name in image_viewers:
        image_viewers[window_name] = ImageViewer(window_name)
    return image_viewers[window_name]


def cv_imshow(window_name, image):
    getImageViewer(window_name).show(image)


class ImageViewer:
    """
     View image and allow to zoom into it by drawing a rectangle or using mouse wheel.
     Also allows to window image using middle mouse button, i.e. re-map channel intensities,
     which is especially useful if image channels have more than 8 bit depth.
    """
    background_color = (64, 64, 0)
    alpha_color = (0, 0, 0)
    text_color = (128, 255, 100)  # (0, 165, 255)
    text_color_outline = (0, 0, 0)
    rectangle_color = text_color
    rectangle_color_outline = text_color_outline
    rectangle_line_width = 1

    window_size = None  # last known window size; if it changes externally, we need to redraw
    image = None  # currently displayed image
    image_dtype_orig = None  # original input image type, may be different from image
    roi = None  # zoomed ROI of image as a rectangle in float-pixel; aspect ratio of ROI may differ from window, in which case border or more image is displayed
    cached_roi_image_all = None  # cached window resized ROI image - performance relevant for large images (e.g. more than screen size) when drawing rectangle
    cached_roi_image_win = None  # cached window resized but before windowing is applied
    cached_roi_image_roi = None  # ROI of cached_roi_image_all - if not the same as roi, cached_roi_image_all and cached_roi_image_win is invalid
    cached_roi_image_windowing = None  # windowing of cached_roi_image_all - if not the same as roi, cached_roi_image_all is invalid
    roi_zoom_step = 0.8  # values < 1.0 reverse zoom direction (hint: 1/0.8 = 1.25)
    roi_min_size = 10  # do not zoom in too much
    zoom_on_mouse_move_max_horizontal_deviation = 0.2
    significant_mouse_move = 5  # to detect if mouse_was_moved

    # windowing: zoom into brightness, i.e. transform image intensities to "window" into a sub-range
    Windowing = namedtuple('Windowing', 'center width', defaults=(0.5, 1.0))
    windowing = Windowing()

    MouseMode = Enum('MouseMode', [
        'is_drawing_rectangle',  # is rectangle drawing in progress?
        'is_zooming',  # is zoom by mouse drag in progress?
        'is_moving',  # is move by mouse drag in progress?
        'is_windowing'])  # is windowing in progress?
    mouse_mode = None

    rectangle = (0, 0, 0, 0)  # last drawn rectangle
    mouse_start_x, mouse_start_y, mouse_was_moved = 0, 0, False  # start point of mouse during mouse move with button down
    mouse_start_roi = None  # start roi during mouse move with button down
    mouse_start_windowing = None  # windowing at start of mouse drag
    mouse_slow_windowing = False
    single_click_zoom_sizes = [0.3, 0.1]  # single click zoom-in factors

    def __init__(self, window_name):
        self.window_name = window_name
        self.create_window()

    def create_window(self):
        cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)
        cv.setMouseCallback(self.window_name, self.on_mouse)

    @property
    def full_size(self):
        return _imageSize(self.image)

    @property
    def full_rectangle(self):
        return (0, 0) + _imageSize(self.image)

    def isWindowVisible(self):
        try:
            return cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) > 0
        except:
            return False

    def show(self, image):
        self.cached_roi_image_all = None
        self.cached_roi_image_win = None
        self.cached_roi_image_roi = None
        self.cached_roi_image_windowing = None

        is_image_copied = False

        # remove transparency channel
        if _imageChannels(image) == 4:
            # Note: for best results alpha channel would be applied after windowing.
            # For simplicity alpha channel is applied now using alpha_color.
            # To avoid unexpected color changes during windowing, alpha_color should be black, white, or grey
            image = _transparencyAsColor(image, self.alpha_color)
            is_image_copied = True

        # store in a supported image type
        self.image_dtype_orig = image.dtype
        if image.dtype == np.int8:  # e.g. cv.resize and cv.threshold fail => display as uint8
            image = image.astype(np.uint8) + 128
            is_image_copied = True
        elif image.dtype == np.int32:
            raise NotImplementedError('Can not display int32 index image.')

        if not is_image_copied:
            image = image.copy()  # always copy input image
            is_image_copied = True

        # The following test patterns were used to check that all of the image is visible at different zoom levels
        # and to show that cv.resize is better at shrinking images than cv.warpAffine (using cv.INTER_AREA and other interpolation methods)
        # and to see windowing artifacts when windowing the shrinked image for fast windowing.
        # # TEST: rainbow border
        # for b, color in enumerate(((0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0))):
        #     cv.rectangle(image, (b, b), (image.shape[1] - 1 - b, image.shape[0] - 1 - b), _color(image, color), 1, cv.LINE_4)
        # # TEST: black/white interference pattern with decreasing frequency
        # for background in (True, False):
        #     for i in range(min(image.shape[:2])):
        #         b = 20 + i * (i + 1) // 2
        #         if b >= min(image.shape[:2]) // 2:
        #             break
        #         color = (0, 0, 0) if background else (255, 255, 255)
        #         cv.rectangle(image, (b, b), (vis.shape[1] - 1 - b, vis.shape[0] - 1 - b), _color(image, color), 2 if background else 1, cv.LINE_4)

        if self.image is None or self.roi is None:
            self.image = image
            self.zoom_out()
        else:
            old_full_rectangle = self.full_rectangle
            old_roi = self.roi
            self.image = image
            relative_roi = _relativeRect(old_roi, old_full_rectangle)
            new_roi = _absoluteRect(relative_roi, self.full_rectangle)
            self.roi = _intersectRects(new_roi, self.full_rectangle)

        self.redraw()

    def redraw(self, do_resize_window=False):
        if self.image is None:
            return

        do_windowing = self.windowing != self.Windowing()
        do_windowing_fast = do_windowing and self.mouse_mode == self.MouseMode.is_windowing
        #if not do_windowing_fast: print('full resolution windowing')

        try:
            window_rect = cv.getWindowImageRect(self.window_name)
        except:
            self.create_window()  # window was closed and needs to be created again
            window_rect = cv.getWindowImageRect(self.window_name)

        if do_resize_window and self.mouse_mode is None:
            aligned_window_rect = _expandRectToAspectRatioInt(window_rect, self.full_size, do_expand=False)
            if window_rect[2:] != aligned_window_rect[2:]:
                cv.resizeWindow(self.window_name, *aligned_window_rect[2:])
                window_rect = cv.getWindowImageRect(self.window_name)

        do_draw = self.mouse_mode == self.MouseMode.is_drawing_rectangle and self.mouse_was_moved

        # print('roi {} {}'.format(self.roi[2] / self.roi[3], self.roi))
        if (self.cached_roi_image_all is not None and
            self.cached_roi_image_roi == self.roi and
            self.cached_roi_image_windowing == self.windowing and
            _imageSize(self.cached_roi_image_all) == window_rect[2:] and
            (self.cached_roi_image_win is None or not do_windowing or do_windowing_fast)  # do not re-use fast windowing for full windowing
           ):
            image = self.cached_roi_image_all
            #print('reuse image')
            if do_draw:
                image = image.copy()
        else:
            text = []
            if not self.is_zoomed_out():
                text.append('{}%'.format(round(100 * min(self.roi[2:]) / min(self.full_size))))  # not the same as zoom_factor

            zoom_factor = min(
                window_rect[2] / self.roi[2],
                window_rect[3] / self.roi[3])
            #print('zoom_factor {} {}'.format(zoom_factor, 1 / zoom_factor))

            # Shrinking the displayed ROI is done first, because cv.resize() gives better quality than cv.warpAffine.
            # Windowing is applied first on the ROI to improve performance for large images
            # and to avoid windowing of background border color.
            # Resulting windowed image format may be different from the original input image.
            win_roi = _clipRectToImage(_growIntRectangle(_expandRectToAspectRatioFloat(self.roi, window_rect[2:], align_sign=-1)), self.image)
            zoom0 = (self.roi[0] - win_roi[0], self.roi[1] - win_roi[1])
            win_image = _roiRectangle(self.image, win_roi)  # image references original image - do copy() before modifying
            if zoom_factor < 1.0 and (do_windowing_fast or not do_windowing):
                if (self.cached_roi_image_win is not None and
                    self.cached_roi_image_roi == self.roi
                   ):
                    win_image = self.cached_roi_image_win  # win_image references cached image - do copy() before modifying
                    #print('reuse win_image')
                else:
                    win_image = _resize(win_image, zoom_factor)
                    self.cached_roi_image_win = win_image  # win_image references cached image - do copy() before modifying
                zoom_factor = 1.0
                zoom0 = (0, 0)
            else:
                self.cached_roi_image_win = None

            if do_windowing:
                win_image = win_image.copy()

                center, width = self.windowing
                cv_range = _range_of_dtype(win_image.dtype)

                def windowingThresh(value):
                    return _range_len(cv_range) * value + cv_range.min

                def windowingThreshRounded(value):
                    value = windowingThresh(value)
                    return _roundInt(value) if cv_range.step == 1 else round(value, 3)

                def windowingDisplayFormat(value):
                    return ('{:.0f}' if cv_range.step == 1 else '{:.3f}').format(value)

                def windowingCenterDisplay(value):
                    value = windowingThresh(value)
                    if self.image_dtype_orig == np.int8 and self.image.dtype == np.uint8:
                        value -= 128
                    else:
                        assert self.image_dtype_orig == self.image.dtype, (self.image_dtype_orig, self.image.dtype)
                    return windowingDisplayFormat(value)

                def windowingWidthDisplay(value):
                    value = _range_len(cv_range) * value
                    return windowingDisplayFormat(value)

                if abs(width) > 1e-8:
                    b = windowingThresh(center - width / 2)
                    # a = 1 / width
                    # win(i) = (i - b) * a + b
                    #        = i*a - b*a + b
                    #        = i * a + b*(1 - a)
                    #        =: i * a + b2
                    # b2 = (1 - a) * b
                    a = 1 / width
                    b2 = (1 - a) * b
                    
                    # image = (image * a + b2).clip(intensity.min, intensity.max).astype(image.dtype) # slow!
                    win_image = _convertAutoScale(win_image, a, b2, cv.CV_8U)
                    text.append('c:{} w:{}'.format(windowingCenterDisplay(center), windowingWidthDisplay(width)))
                    #print('c:{} w:{}'.format(windowingThreshRounded(center), windowingThreshRounded(width)))
                else:
                    b = windowingThreshRounded(center)
                    win_image = _thresholdAsUint8(win_image, b)
                    text.append('c:{}'.format(windowingCenterDisplay(center)))
                    #print('c:{}'.format(b))

                if zoom_factor < 1.0 and not do_windowing_fast:
                    win_image = _resize(win_image, zoom_factor)
                    zoom_factor = 1.0
                    zoom0 = (0, 0)

            # make image uint8
            if win_image.dtype != np.uint8:
                # cv.warpAffine works well for uint8 3-channel images, but can not transform image types.
                # Also some drawing functions like cv.putText work best on uint8.
                win_image = _convertAutoScale(win_image, 1, 0, cv.CV_8U)
                assert win_image.dtype == np.uint8

            # make image 3 channels
            if _imageChannels(win_image) == 1:
                win_image = cv.cvtColor(win_image, cv.COLOR_GRAY2BGR)
            # exotic cases like transparency should be handled earlier, e.g. converted in self.show()
            assert _imageChannels(win_image) == 3, 'unsupported number of channels: {}'.format(_imageChannels(win_image))

            if self.cached_roi_image_all is not None and _imageSize(self.cached_roi_image_all) == window_rect[2:]:
                image = self.cached_roi_image_all  # re-use memory of old image
                #print('reuse memory')
            else:
                image = np.empty((window_rect[3], window_rect[2], 3), np.uint8)
                #print('new memory')

            #print('zoom0 {}'.format(zoom0))
            M = np.asarray(((zoom_factor, 0, (0.5 - zoom0[0]) * zoom_factor),
                            (0, zoom_factor, (0.5 - zoom0[1]) * zoom_factor)), np.double)
            interpolation = cv.INTER_AREA if zoom_factor < 1 else cv.INTER_NEAREST
            assert win_image.dtype == image.dtype, (win_image.dtype, image.dtype)
            cv.warpAffine(win_image, M, _imageSize(image), image, interpolation, cv.BORDER_CONSTANT, _color(image, self.background_color))

            if text:
                text = ' '.join(text)
                pos = (5, 17)
                fontFace = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 0.4
                cv.putText(image, text, pos, fontFace, fontScale, _color(image, self.text_color_outline), 2, cv.LINE_AA)
                cv.putText(image, text, pos, fontFace, fontScale, _color(image, self.text_color), 1, cv.LINE_AA)

            self.cached_roi_image_all = image
            self.cached_roi_image_roi = self.roi
            self.cached_roi_image_windowing = self.windowing

        if do_draw:
            thick = self.rectangle_line_width
            pt1, pt2 = _rectPoints(self.rectangle)
            cv.rectangle(image, pt1, pt2, _color(image, self.rectangle_color_outline), thick * 2)
            cv.rectangle(image, pt1, pt2, _color(image, self.rectangle_color), thick)

        if not self.isWindowVisible():
            cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)

        self.window_size = window_rect[2:]
        cv.imshow(self.window_name, image)

    def window_rectangle_to_roi(self, rectangle, roi=None):
        """
        Convert rectangle in window coordinates to ROI coordinates.
        Window coordinates are used by mouse events.
        ROI coordinates are image coordinates offset by ROI position.
        Beyond zoom factor this transformation also considers extra parts outside of ROI
        that are displayed in the window, i.e. more of the image or some border area.
        If ROI was changed, the original ROI can be given as an argument.
        If the window was resized externally (e.g. manually),
        it should be re-drawn before using this transformation.
        """
        if roi is None:
            roi = self.roi
        f = max(roi[2] / self.window_size[0],
                roi[3] / self.window_size[1])
        return (rectangle[0] * f, rectangle[1] * f, rectangle[2] * f, rectangle[3] * f)

    def window_point_to_roi(self, point, roi=None):
        return self.window_rectangle_to_roi(point + (0, 0), roi)[:2]

    def window_vector_to_roi(self, vector, roi=None):
        return self.window_rectangle_to_roi((0, 0) + vector, roi)[2:]

    def zoom_out(self):
        self.roi = self.full_rectangle

    def is_zoomed_out(self):
        return _roundIntRect(self.roi) == self.full_rectangle

    def set_sanitized_roi(self, roi):
        roi = _expandRectToAspectRatioFloat(roi, self.full_size)
        if max(roi[2:]) < self.roi_min_size:
            roi = _growRectAroundCenter(roi, self.roi_min_size / max(roi[2:]))
        roi = _moveRectIntoImage(roi, *self.full_size)
        roi = _intersectRects(roi, self.full_rectangle)
        self.roi = roi

    def zoom_to_rectangle(self, rectangle):
        roi = (self.roi[0] + rectangle[0], self.roi[1] + rectangle[1], rectangle[2], rectangle[3])
        roi = _expandRectToAspectRatioFloat(roi, self.window_size)
        roi = _expandRectToAspectRatioFloat(roi, self.full_size, do_expand=False, align_sign=-1)
        self.set_sanitized_roi(roi)

    def zoom_to_point(self, old_point, old_roi, zoom_factor):
        min_zoom_factor = self.roi_min_size / max(old_roi[2:])
        if zoom_factor < min_zoom_factor:
            zoom_factor = min_zoom_factor
        x, y = old_point
        roi = (old_roi[0] + x * (1 - zoom_factor),
               old_roi[1] + y * (1 - zoom_factor),
               old_roi[2] * zoom_factor,
               old_roi[3] * zoom_factor)
        self.set_sanitized_roi(roi)

    def zoom_to_point_jump(self, old_point, old_roi, jump_factor):
        if jump_factor == 0:
            self.roi = old_roi
            return

        if jump_factor < 0:
            min_zoom_factor = self.roi_min_size / max(old_roi[2:])
            max_zoom_factor = 1.0
            zoom_factor = max_zoom_factor + (max_zoom_factor - min_zoom_factor) * jump_factor
        else:
            min_zoom_factor = 1.0
            max_zoom_factor = max(self.full_size[0] / old_roi[2], self.full_size[1] / old_roi[3])
            zoom_factor = min_zoom_factor + (max_zoom_factor - min_zoom_factor) * jump_factor

        self.zoom_to_point(old_point, old_roi, zoom_factor)

    def move_roi(self, old_roi, shift_pixel):
        roi = (old_roi[0] - shift_pixel[0], old_roi[1] - shift_pixel[1], old_roi[2], old_roi[3])
        roi = _moveRectIntoImage(roi, *self.full_size)
        roi = _intersectRects(roi, self.full_rectangle)
        self.roi = roi

    def update_rectangle_on_mouse_move_to(self, x, y, is_start=False):
        if is_start:
            self.mouse_start_x, self.mouse_start_y, self.mouse_was_moved = x, y, False
            self.rectangle = (x, y, 0, 0)
            return

        self.rectangle = (min(self.mouse_start_x, x),
                          min(self.mouse_start_y, y),
                          abs(self.mouse_start_x - x),
                          abs(self.mouse_start_y - y))

    def update_zoom_on_mouse_move_to(self, x, y, is_start=False):

        if is_start:
            self.mouse_start_x, self.mouse_start_y, self.mouse_was_moved = x, y, False
            self.mouse_start_roi = self.roi
            return

        old_roi = self.mouse_start_roi

        shift = (x - self.mouse_start_x, y - self.mouse_start_y)
        if abs(shift[0]) > self.window_size[0] * self.zoom_on_mouse_move_max_horizontal_deviation:
            jump_factor = 0.0
        elif shift[1] < 0:
            jump_factor = (shift[1] - 1) / (self.mouse_start_y + 1)
        elif shift[1] > 0:
            jump_factor = (shift[1] + 1) / (self.window_size[1] - self.mouse_start_y)
        else:
            jump_factor = 0.0

        old_point = self.window_point_to_roi((self.mouse_start_x, self.mouse_start_y), old_roi)
        self.zoom_to_point_jump(old_point, old_roi, jump_factor)

    def update_move_on_mouse_move_to(self, x, y, is_start=False):
        if is_start:
            self.mouse_start_x, self.mouse_start_y, self.mouse_was_moved = x, y, False
            self.mouse_start_roi = self.roi
            return

        shift = self.window_vector_to_roi((x - self.mouse_start_x, y - self.mouse_start_y))
        self.move_roi(self.mouse_start_roi, shift)

    def update_windowing_on_mouse_move_to(self, x, y, is_start=False, is_shift_pressed=False):

        if is_start:
            self.mouse_start_x, self.mouse_start_y, self.mouse_was_moved = x, y, False
            self.mouse_start_windowing = self.windowing
            self.mouse_slow_windowing = is_shift_pressed
            return

        is_high_dynamic_range = self.image.dtype not in (np.uint8, np.int8)

        shift = (x - self.mouse_start_x, y - self.mouse_start_y)
        border = 8
        shift_range = (max(border, self.mouse_start_x if shift[0] < 0 else self.window_size[0] - 1 - self.mouse_start_x),
                       max(border, self.mouse_start_y if shift[1] < 0 else self.window_size[1] - 1 - self.mouse_start_y))
        factor = (shift[0] / shift_range[0], shift[1] / shift_range[1])
        if self.mouse_slow_windowing:
            slow = 0.01 if is_high_dynamic_range else 0.1
            factor = (factor[0] * slow, factor[1] * slow)
        # print('factor {}'.format(factor))

        old_center, old_width = self.mouse_start_windowing

        if is_high_dynamic_range or self.mouse_slow_windowing:
            new_width = old_width + factor[0]
            new_center = old_center - factor[1]
            self.windowing = self.Windowing(new_center, max(0.0, new_width))
        else:
            new_width = old_width + (1 - old_width) * factor[0] if factor[0] > 0 else old_width + old_width * factor[0]
            new_center = old_center - (1 - old_center) * factor[1] if factor[1] < 0 else old_center - old_center * factor[1]
            self.windowing = self.Windowing(min(max(new_center, 0.0), 1.0), min(max(new_width, 0.0), 1.0))

    def on_mouse(self, event, x, y, flags, param):
        """
        Mouse Usage:
        ============

        Left mouse does everything important:
            * click zooms in/out using default zoom value 1
            * click on border resizes window to remove border
            If zoomed out:
            * drag does zoom via rectangle and updates default zoom value 1
            If zoomed in:
            * drag does move the view

        Right mouse does the same, except it never drag moves:
            * click zooms in/out using default zoom value 2
            * click on border resizes window to remove border
            * drag does zoom via rectangle and updates default zoom value 2

        Middle mouse:
            * drag does windowing; shift-drag for fine-tuning
            * click resets windowing
            * control-drag zooms in/out around start point depending on vertical position

        Mouse wheel:
            zooms in/out at mouse position
        """

        # known_flags = (cv.EVENT_FLAG_LBUTTON | cv.EVENT_FLAG_RBUTTON | cv.EVENT_FLAG_MBUTTON |
        #                cv.EVENT_FLAG_CTRLKEY | cv.EVENT_FLAG_SHIFTKEY | cv.EVENT_FLAG_ALTKEY)
        # print('mouse {}, {}, {}, {:b} {}, {}'.format(event, x, y, flags & known_flags, flags, param))

        if self.window_size != cv.getWindowImageRect(self.window_name)[2:]:
            self.mouse_mode = None
            self.redraw()  # brute force - seems to work best (note: should be called in correct thread)

        if max(abs(x - self.mouse_start_x), abs(y - self.mouse_start_y)) >= self.significant_mouse_move:
            self.mouse_was_moved = True

        def assert_mouse_mode(mouse_mode):
            if self.mouse_mode == mouse_mode:
                return True
            if self.mouse_mode == self.MouseMode.is_windowing:
                self.windowing = self.mouse_start_windowing  # restore value before operation
            elif self.mouse_mode == self.MouseMode.is_zooming:
                self.roi = self.mouse_start_roi  # restore value before operation
            self.mouse_mode = None
            self.mouse_was_moved = True  # cancels a click operation
            self.redraw()
            return False

        def start_drawing_rectangle(x, y):
            if not assert_mouse_mode(None):
                return

            self.mouse_mode = self.MouseMode.is_drawing_rectangle
            self.update_rectangle_on_mouse_move_to(x, y, True)
            self.redraw()

        def continue_drawing_rectangle(x, y):
            if assert_mouse_mode(self.MouseMode.is_drawing_rectangle):
                self.update_rectangle_on_mouse_move_to(x, y)
                self.redraw()

        def end_drawing_rectangle(x, y):
            if not assert_mouse_mode(self.MouseMode.is_drawing_rectangle):
                return False
            self.mouse_mode = None
            self.update_rectangle_on_mouse_move_to(x, y)
            self.zoom_to_rectangle(self.window_rectangle_to_roi(self.rectangle))
            self.redraw()
            return True

        def start_moving(x, y):
            if not assert_mouse_mode(None):
                return

            self.mouse_mode = self.MouseMode.is_moving
            self.update_move_on_mouse_move_to(x, y, True)
            self.redraw()

        def continue_moving(x, y):
            if assert_mouse_mode(self.MouseMode.is_moving):
                self.update_move_on_mouse_move_to(x, y)
                self.redraw()

        def end_moving(x, y):
            if assert_mouse_mode(self.MouseMode.is_moving):
                self.mouse_mode = None
                self.update_move_on_mouse_move_to(x, y)
                self.redraw()

        def click_on_border_resizes(x, y):
            p = self.window_point_to_roi((x, y))
            if (p[0] + self.roi[0] < self.full_size[0] and
                    p[1] + self.roi[1] < self.full_size[1]):
                return False

            self.mouse_mode = None
            self.redraw(do_resize_window=True)
            return True

        def zoom_out(do_resize_window=False):
            self.mouse_mode = None
            self.zoom_out()
            self.redraw(do_resize_window)

        def zoom_in(x, y, button_index):
            self.mouse_mode = None
            self.zoom_to_point(self.window_point_to_roi((x, y)), self.roi, self.single_click_zoom_sizes[button_index])
            self.redraw()

        def set_default_zoom(button_index):
            self.single_click_zoom_sizes[button_index] = max(self.roi[2:]) / max(self.full_size)

        ###############
        # left mouse
        ###############

        if event == cv.EVENT_LBUTTONDOWN:
            if self.is_zoomed_out():
                start_drawing_rectangle(x, y)
            else:
                start_moving(x, y)

        elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON) != 0:
            if self.mouse_was_moved:
                if self.is_zoomed_out():
                    continue_drawing_rectangle(x, y)
                else:
                    continue_moving(x, y)

        elif event == cv.EVENT_LBUTTONUP:
            if self.mouse_was_moved:
                if self.is_zoomed_out():
                    if (end_drawing_rectangle(x, y)):
                        set_default_zoom(0)
                else:
                    end_moving(x, y)
            elif not click_on_border_resizes(x, y):
                if self.is_zoomed_out():
                    zoom_in(x, y, 0)
                else:
                    zoom_out()

        elif event == cv.EVENT_LBUTTONDBLCLK:
            pass


        ###############
        # right mouse
        ###############

        elif event == cv.EVENT_RBUTTONDOWN:
            start_drawing_rectangle(x, y)

        elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_RBUTTON) != 0:
            if self.mouse_was_moved:
                continue_drawing_rectangle(x, y)

        elif event == cv.EVENT_RBUTTONUP:
            if self.mouse_was_moved:
                if (end_drawing_rectangle(x, y)):
                    set_default_zoom(1)
            elif not click_on_border_resizes(x, y):
                if self.is_zoomed_out():
                    zoom_in(x, y, 1)
                else:
                    zoom_out()

        elif event == cv.EVENT_RBUTTONDBLCLK:
            if self.mouse_was_moved:
                zoom_out(do_resize_window=True)

        ###############
        # middle mouse
        ###############

        if event == cv.EVENT_MBUTTONDOWN:
            if (flags & cv.EVENT_FLAG_CTRLKEY) == 0:
                if assert_mouse_mode(None):
                    self.mouse_mode = self.MouseMode.is_windowing
                    self.update_windowing_on_mouse_move_to(x, y, True, (flags & cv.EVENT_FLAG_SHIFTKEY) != 0)
                    self.redraw()
            else:
                if assert_mouse_mode(None):
                    self.mouse_mode = self.MouseMode.is_zooming
                    self.update_zoom_on_mouse_move_to(x, y, True)
                    self.redraw()

        elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_MBUTTON) != 0:
            if (flags & cv.EVENT_FLAG_CTRLKEY) == 0:
                if assert_mouse_mode(self.MouseMode.is_windowing) and self.mouse_was_moved:
                    self.update_windowing_on_mouse_move_to(x, y)
                    self.redraw()
            else:
                if assert_mouse_mode(self.MouseMode.is_zooming):
                    self.update_zoom_on_mouse_move_to(x, y)
                    self.redraw()

        elif event == cv.EVENT_MBUTTONUP:
            if (flags & cv.EVENT_FLAG_CTRLKEY) == 0:
                if assert_mouse_mode(self.MouseMode.is_windowing):
                    self.mouse_mode = None
                    if self.mouse_was_moved:
                        self.update_windowing_on_mouse_move_to(x, y)
                    else:
                        self.windowing = self.Windowing()
                    self.redraw()
            else:
                if assert_mouse_mode(self.MouseMode.is_zooming):
                    self.mouse_mode = None
                    self.update_zoom_on_mouse_move_to(x, y)
                    self.redraw()

        elif event == cv.EVENT_MBUTTONDBLCLK:
            pass


        ###############
        # mouse wheel
        ###############

        elif event == cv.EVENT_MOUSEWHEEL:
            if self.mouse_mode is None:  # not assert_mouse_mode(None) because wheel may be triggered by accident
                do_zoom_positive = flags > 0
                zoom_factor = self.roi_zoom_step if do_zoom_positive else 1 / self.roi_zoom_step

                self.zoom_to_point(self.window_point_to_roi((x, y)), self.roi, zoom_factor)
                self.redraw()


_cv_type_of_np_types = {
    np.uint8: cv.CV_8U,
    np.int8: cv.CV_8S,
    np.uint16: cv.CV_16U,
    np.int16: cv.CV_16S,
    np.int32: cv.CV_32S,
    np.float32: cv.CV_32F,
    np.float64: cv.CV_64F,
}

_np_type_of_cv_type = {v: k for k, v in _cv_type_of_np_types.items()}

_CvTypeRange = namedtuple('_CvTypeRange', 'min max step')

_range_of_cv_types = {
    cv.CV_8U: _CvTypeRange(0, 255, 1),
    cv.CV_8S: _CvTypeRange(-128, 127, 1),
    cv.CV_16U: _CvTypeRange(0, 65535, 1),
    cv.CV_16S: _CvTypeRange(-32768, 32767, 1),
    cv.CV_32S: _CvTypeRange(-2147483648, 2147483647, 1),  # usually an index array, no color defined
    cv.CV_32F: _CvTypeRange(0.0, 1.0, 0.0),
    cv.CV_64F: _CvTypeRange(0.0, 1.0, 0.0),
}

def _range_of_dtype(dtype):
    return _range_of_cv_types[_cv_type_of_np_types[dtype.type]]

def _range_len(range):
    return range.max - range.min + range.step


def _convertAutoScale(src, alpha=1.0, beta=0.0, cv_type=None):
    """
    Since cv::Mat::convertTo is not available in Python, we use cv.addWeighted as a workaround.
    See https://github.com/opencv/opencv/issues/7231 "bring back cv::convertScale".
    Does automatic range conversion, where alpha and beta are considered to be in range of src.
    cv_type is an OpenCV data type like cv.CV_8U or None if data type stays unchanged.
    If there is no data type change, conversion is done in-place, i.e. src is modified.
    """
    dtype = _np_type_of_cv_type[cv_type] if cv_type is not None else src.dtype
    if src.dtype == dtype:
        dst = src
    else:
        dst = np.empty(src.shape, dtype)
        # do automatic range conversion
        if cv_type == cv.CV_32S:
            raise NotImplementedError('No definition of color for cv.CV_32S')
        src_range = _range_of_dtype(src.dtype)
        dst_range = _range_of_cv_types[cv_type]
        scale = _range_len(dst_range) / _range_len(src_range)
        # f1(x) = x * alpha + beta
        # f2(x) = (x - src_range.min) * scale + dst_range.min
        # f2(f1(x)) = ((x * alpha + beta) - src_range.min) * scale + dst_range.min
        #           = x * (alpha * scale) + (beta - src_range.min) * scale + dst_range.min
        alpha *= scale
        beta = (beta - src_range.min) * scale + dst_range.min

    cv.addWeighted(src, alpha, src, 0, beta, dst, cv_type)
    return dst


def _thresholdAsUint8(image, b):
    """
    Conversion is done in-place, i.e. image is modified.
    """
    cv_range = _range_of_dtype(image.dtype)

    # documentation of cv.threshold says "8-bit or 32-bit floating point", but it also works for 16-bit int and 64-bit float, but not for 8-bit signed int.
    if image.dtype in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        cv.threshold(image, b - cv_range.step, 255, cv.THRESH_BINARY, image)
    else:
        # image = ((image >= b) * (intensity.max - intensity.min) + intensity.min).astype(image.dtype)  # slow!
        # np.choose(image >= b, np.array((intensity.min, intensity.max), image.dtype), image, 'clip')  # slow!
        # image = np.where(image >= b, *np.array((intensity.max, intensity.min), image.dtype))  # a little less slow
        # fastest, but still a lot slower than cv.threshold:
        image[image < b] = 0
        image[image >= b] = 255  # for np.int8 the value is silently cast to -1 and back to 255 in image.astype(np.uint8)

    return image if image.dtype == np.uint8 else image.astype(np.uint8)


def _resize(win_image, zoom_factor):
    return cv.resize(win_image,
                     (_ceilInt(win_image.shape[1] * zoom_factor),
                      _ceilInt(win_image.shape[0] * zoom_factor)),
                     fx=zoom_factor, fy=zoom_factor,
                     interpolation=cv.INTER_AREA)


def _color(image, color):
    if image.dtype == np.uint8: return color
    if image.dtype == np.int8: return tuple(v - 128 for v in color)
    if image.dtype in (np.float32, np.float64): return tuple(v / 255 for v in color)
    if image.dtype == np.uint16: return tuple(v * 257 for v in color)
    if image.dtype == np.int16: return tuple(v * 257 - 32768 for v in color)
    raise NotImplementedError('No definition of color for {}'.format(image.dtype.type))


def _transparencyAsColor(image, background_color):
    cv_range = _range_of_dtype(image.dtype)
    foreground = image[:, :, :3].astype(float)
    background = np.array(_color(image, background_color), float).reshape((1, 1, 3))
    alpha = (image[:, :, 3:].astype(float) - cv_range.min) / (cv_range.max - cv_range.min)
    return (foreground * alpha + background * (1 - alpha)).astype(image.dtype)


def _imageChannels(image):
    return image.shape[2] if len(image.shape) > 2 else 1


def _imageSize(image):
    return image.shape[1::-1]


def _roiRectangle(image, rect):
    return image[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]


def _floorInt(value):
    return int(math.floor(value))


def _ceilInt(value):
    return int(math.ceil(value))


def _roundInt(value):
    return int(round(value))


def _roundIntRect(rectangle):
    x1 = _roundInt(rectangle[0])
    y1 = _roundInt(rectangle[1])
    x2 = _roundInt(rectangle[0] + rectangle[2])
    y2 = _roundInt(rectangle[1] + rectangle[3])
    return (x1, y1, x2 - x1, y2 - y1)


def _growIntRectangle(rectangle):
    x1 = _floorInt(rectangle[0])
    y1 = _floorInt(rectangle[1])
    x2 = _ceilInt(rectangle[0] + rectangle[2])
    y2 = _ceilInt(rectangle[1] + rectangle[3])
    return (x1, y1, x2 - x1, y2 - y1)


def _absoluteRect(relative_rectangle, full_rectangle):
    bx, by, bw, bh = full_rectangle
    rx, ry, rw, rh = relative_rectangle
    return (bx + bw * rx,
            by + bh * ry,
            bw * rw,
            bh * rh)


def _relativeRect(absolute_rectangle, full_rectangle):
    rx, ry, rw, rh = absolute_rectangle
    bx, by, bw, bh = full_rectangle
    return ((rx - bx) / bw,
            (ry - by) / bh,
            rw / bw,
            rh / bh)


def _expandRectToAspectRatioFloat(rectangle, aspect_size, do_expand = True, align_sign = 0):
    r_x, r_y, r_width, r_height = rectangle
    aspect_size_width, aspect_size_height = aspect_size

    if (aspect_size_width * r_height < r_width * aspect_size_height) == do_expand:
        new_height = r_width * aspect_size_height / aspect_size_width
        if align_sign >= 0:
            r_y += (r_height - new_height) * (0.5 if align_sign == 0 else 1.0)
        r_height = new_height
    else:
        new_width = aspect_size_width * r_height / aspect_size_height
        if align_sign >= 0:
            r_x += (r_width - new_width) * (0.5 if align_sign == 0 else 1.0)
        r_width = new_width
    return (r_x, r_y, r_width, r_height)


def _expandRectToAspectRatioInt(rectangle, aspect_size, do_expand = True, align_sign = 0):
    return _roundIntRect(_expandRectToAspectRatioFloat(rectangle, aspect_size, do_expand, align_sign))


def _rectPoints(r):
    return (r[0], r[1]), (r[0] + r[2], r[1] + r[3])


def _isRectValid(r):
    return r is not None and r[2] >= 0 and r[3] >= 0


def _intersectRects(r1, r2):
    if not _isRectValid(r1) or not _isRectValid(r2):
        return None

    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])
    if x1 > x2 or y1 > y2:
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def _clipRectToImage(rect, image):
    return _intersectRects(rect, (0, 0, image.shape[1], image.shape[0]))


def _moveRectIntoImage(rect, image_width, image_height):
    x, y, width, height = rect
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= image_width - width:
        x = image_width - width
    if y >= image_height - height:
        y = image_height - height
    if x < 0:
        x = x / 2  # center horizontally in image, both sides out of image (x stays negative)
    if y < 0:
        y = y / 2  # center vertically in image, both sides out of image (y stays negative)

    return (x, y, width, height)


def _resizeRectAroundCenter(rectangle, size):
    return (rectangle[0] + 0.5 * (rectangle[2] - size[0]),
            rectangle[1] + 0.5 * (rectangle[3] - size[1]),
            size[0],
            size[1])


def _growRectAroundCenter(rectangle, factor):
    return _resizeRectAroundCenter(rectangle, (rectangle[2] * factor, rectangle[3] * factor))



print('{}:\n{}{}'.format(ImageViewer.__name__, ImageViewer.__doc__, ImageViewer.on_mouse.__doc__))


if __name__ == '__main__':
    import sys
    image = cv.imread(sys.argv[1] if len(sys.argv) == 2 else 'example.png', cv.IMREAD_COLOR)

    zoom_steps = 10
    zoom_step = -1

    rotate_steps = 90
    rotate_step = -1

    types = (np.uint8, np.int8, np.uint16, np.int16, np.float32, np.float64)  # not np.int32
    channels = (3, 1, 4)  # BGR, grey, BGRA
    formats = [(t, c) for t in types for c in channels]
    format_step = -1


    def rotate_image_grow_size(img, angle):
        size = np.array(_imageSize(img))
        M = cv.getRotationMatrix2D(tuple(size / 2.), angle, 1.)
        MM = np.absolute(M[:, :2])
        size_new = MM @ size
        M[:, -1] += (size_new - size) / 2.
        return cv.warpAffine(img, M, tuple(size_new.astype(int)), borderMode=cv.BORDER_REPLICATE, flags=cv.INTER_CUBIC)


    interactive_window = getImageViewer("ImageViewer")
    while True:
        zoom_step = (zoom_step + 1) % zoom_steps
        zoom = (abs(zoom_step - zoom_steps / 2) + zoom_steps / 2) / zoom_steps
        print('zoom={}'.format(zoom))

        rotate_step = (rotate_step + 1) % rotate_steps
        angle = 360 * (rotate_step / rotate_steps)
        print('angle={}'.format(angle))

        format_step = (format_step + 1) % len(formats)
        image_type, channels_count = formats[format_step]
        print('format {} ({} x {})'.format(format_step, image_type.__name__, channels_count))

        vis = cv.resize(rotate_image_grow_size(image.copy(), angle), None, fx=zoom, fy=zoom, interpolation=cv.INTER_AREA)
        zoom2 = 2.0  # super sized
        vis = cv.resize(vis, None, fx=zoom2, fy=zoom2, interpolation=cv.INTER_AREA)

        if channels_count not in (1, 3, 4):
            raise NotImplementedError()
        if channels_count == 1:
            vis = cv.cvtColor(vis, cv.COLOR_BGR2GRAY)

        if image_type == np.uint8:
            pass
        elif image_type == np.int8:
            vis = (vis.astype(np.int16) - 128).astype(np.int8)
        elif image_type == np.uint16:
            vis = vis.astype(np.uint16) * 257
        elif image_type == np.int16:
            vis = (vis.astype(np.int32) * 257 - 32768).astype(np.int16)
        elif image_type in (np.float32, np.float64):
            vis = vis.astype(image_type) / 255
        else:
            raise NotImplementedError()

        # test: use original image values in lower/right quadrant - they can be viewed using windowing
        if vis.dtype in (np.uint16, np.int16):
            w, h = _imageSize(vis)
            if len(vis.shape) == 2:
                vis[h//2:, w//2:] %= 256
            else:
                vis[h//2:, w//2:, :3] %= 256

        if channels_count == 4:
            pad = ((0, 0), (0, 0), (0, 1))
            cv_range = _range_of_dtype(vis.dtype)
            opaque_value = cv_range.max
            transparent_value = cv_range.min
            vis = np.pad(vis, pad, 'constant', constant_values=opaque_value)
            w, h = _imageSize(vis)
            vis[:w//2, :h//2, 3] = transparent_value + (opaque_value - transparent_value) / 2
            vis[:w//4, :h//4, 3] = transparent_value

        interactive_window.show(vis)
        key = cv.waitKey()
        print('key={}'.format(key))
        if key < 0 or key in (27,):
            break
