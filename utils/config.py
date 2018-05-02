from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/data/VOC/VOCdevkit/VOC2007/'
    train_image_dir = '/data/tianchi/image_9000'
    train_txt_dir = '/data/tianchi/txt_9000'
    val_image_dir = '/data/tianchi/image_1000'
    val_txt_dir = '/data/tianchi/txt_1000'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    # rpn_sigma = 3.
    # roi_sigma = 1.
    rpn_sigma = 1
    roi_sigma = .3


    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0010
    #lr_decay = 0.1  # 1e-3 -> 1e-4
    lr_decay = 0.3  
    lr_decay_step = 500 # every 1000 step lr decay
    lr = 1e-4
    save_loops = 1000# save every 1000 loops


    # anchor boxs
    ratios = [1.44780479,   4.3442481,   8.86579222,  17.65785927, 44.89712161]

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training 
    epoch = 14


    use_adam =  True# Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16-caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
