from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', default='', type=str, help='the results dir, default is expr_dir/results  ')
        self.parser.add_argument('--n_samples', type=int, default=4, help='#samples for multimodal')
        self.parser.add_argument('--how_many', type=int, default=100, help='how many test images to run')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.is_train = False
