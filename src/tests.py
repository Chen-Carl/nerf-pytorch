from model import init_models, nerf_forward

def testForward():
    model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()


testForward()