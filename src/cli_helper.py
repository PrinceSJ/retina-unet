
import sys

def parse_opts():
    help_text = 'usage: python run_experiments.py [args] \n\
    This script triggers training and testing with different setups. \n\
    available datasets:                 DRIVE, Synth \n\
    available architectures:            resnet, unet \n\
    \n\
    --help, -h                          show this helptext\n\
    --testsets, -p [array of datasets]  define datasets to test with\n\
    --trainsets, -t [array of datasets] define datasets to train on\n\
    --finetune, -f  [array of datasets] define datasets to finetune on \n\
    --only-testing                      only run prediciton \n\
    --only-training                     only run trainig    \n\
    --archs, --architectures, -a [array of architectures] archs to use\n\
    \n\
    \n\
    --finetune, -f\n\
    This option enables finetuning on disered datasets. The same order as trainsets \n\
    are used. If no finetuning should be applied enter _. Here an example:\n\
    \n\
    python run_experiments.py -t DRIVE Synth -f _ DRIVE -a unet\n\
    \n\
    This would train an unet on DRIVE data without finetuning and an unet on Synth data\n\
    with finetuning on DRIVE data.\n\
    '

    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        print(help_text)
        exit()

    currKey = ''
    testsets = []
    trainsets = []
    finetune = []
    archs = []
    only_training = False
    only_testing = False

    for arg in sys.argv[1:]:
        if arg[:1] == '-':
            if arg[1:] == 'p' or arg[2:] == 'testsets':
                currKey = 'testsets'
            elif arg[1:] == 't' or arg[2:] == 'trainsets':
                currKey = 'trainsets'
            elif arg[1:] == 'f' or arg[2:] == 'finetune':
                currKey = 'finetune'
            elif arg[1:] == 'a' or arg[2:] == 'architectures' or arg[2:] == 'archs':
                currKey = 'archs'
            elif arg[2:] == 'only-training':
                only_training = True
            elif arg[2:] == 'only-testing':
                only_testing = True
        elif not currKey == '':
            eval(currKey).append(arg)
    
    if len(trainsets) == 0:
        trainsets = ['DRIVE', 'Synth']
    if len(testsets) == 0:
        testsets = ['DRIVE', 'Synth']
    if len(archs) == 0:
        archs = ['resnet', 'unet']

    for i in range(len(trainsets)):
        if finetune[i] == '_':
            finetune[i] = None
        if not finetune[i]:
            finetune.append(None)
    
    assert(len(trainsets) == len(finetune))

    return trainsets, testsets, finetune, archs, only_training, only_testing

