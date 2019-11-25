from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # log&checkpoint
        self.parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--display_id', type=int, default=0, help='visdom id')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom id')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')
        # lr scheduler and criterion
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--content_gan_mode', default='lsgan', help='default implementation:[dcgan, lsgam, hinge]')
        self.parser.add_argument('--image_gan_mode', default='hinge', help='default implementation:[dcgan, lsgam, hinge]')
        self.parser.add_argument('--rec_mode', default='l1', help='default implementation:[l1, mse]')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # lambda parameters
        ## optional parameters
        self.parser.add_argument('--lambda_cyc', type=float, default=10, help='weight for cycle consistency (It is not required, but it can improve training stability)')
        self.parser.add_argument('--lambda_div', type=float, default=0.05, help='weight for diversity regulazation (It is not required, but it can improve the visual diversity)')
        ## requisite parameters
        self.parser.add_argument('--lambda_style', type=float, default=1, help='weight for reconstruction of style')
        self.parser.add_argument('--lambda_content', type=float, default=1, help='weight for reconstruction of content')
        self.parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10, help='weight for identity consistency')
        self.is_train = True
