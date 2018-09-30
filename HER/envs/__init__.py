from gym.envs.registration import register
import HER

register(
    id='pusher-v0',
    entry_point='HER.envs.pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='pusher-v1',
    entry_point='HER.envs.close_pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='oc_pusher-v0',
    entry_point='HER.envs.oc_pusher:BaxterEnv',
    kwargs={'max_len':20}
)


register(
    id='img_pusher-v0',
    entry_point='HER.envs.oc_pusher:BaxterEnv',
    kwargs={'max_len':20, 'img': True}
)

register(
    id='bb_pusher-v0',
    entry_point='HER.envs.bb_pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='bb_pusher-v1',
    entry_point='HER.envs.bb_pusher:BaxterEnv',
    kwargs={'max_len':20, 'bbox_noise': 2.0}
)

register(
    id='fakercnn_pusher-v0',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'max_len':20, 'bbox_noise': 0.0}
)

register(
    id='img_pusher-v1',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'max_len':20, 'img': True}
)

register(
    id='fakercnn_pusher-v1',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml'}
)

register(
    id='fakercnn_pusher-v2',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'bbox_noise': 1.5}
)

#nothing to do here... just debugging, static camera
register(
    id='active_pusher-v0',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0}
)

register(
    id='active_pusher-v1',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0, 'rot': True, 'rot_scale': 0.0}
)

#camera can move bigly
register(
    id='active_pusher-v2',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={}
)

#camera can move bigly
register(
    id='active_pusher-v5',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'rot': True}
)


#camera can move and active vision is necessary!
register(
    id='active_pusher-v3',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml'}
)

register(
    id='active_pusher-v30',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'bbox_noise': 0.5}
)

#handicapped distractor
register(
    id='active_pusher-v4',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'pos_scale': 0.0}
)

#with auxiliary rwd
register(
    id='active_pusher-v6',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'aux_rwd': True}
)

register(
    id='active_pusher-v7',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'aux_rwd': True, 'pos_scale': 0.0}
)

register(
    id='active_pusher-v8',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'aux_rwd': False, 'randcam': True}
)

#rcnn trajs
register(
    id='rcnn_pusher-v0',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'test': True}
)
register(
    id='rcnn_pusher-v1',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'pos_scale': 0.0, 'test': True}
)

register(
    id='rcnn_pusher-v2',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'test': True, 'randcam': True}
)

#like active_pusher but img only
register(
    id='imgonly_pusher-v0',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0, 'not_oc': True}
)

register(
    id='imgonly_pusher-v1',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0, 'not_oc': True, 'no_obj': True}
)
