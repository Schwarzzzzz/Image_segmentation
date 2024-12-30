from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=9999, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        parser.add_argument('-s', type=str, help='IMG_source_path')
        parser.add_argument('-d', type=str, help='IMG_save_path')
        parser.add_argument('-r', type=int, default=0, help='resized_height')
        parser.add_argument('-m', type=int, help='out_img_split_row')
        parser.add_argument('-n', type=int, help='out_img_split_col')
        parser.add_argument('-spatial', type=int, default=0, help='calculate spatial_information: 1 for cutted, 2 for mini square')

        self.isTrain = False
        return parser
