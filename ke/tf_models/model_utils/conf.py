import tensorflow as tf

session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6,
                              allow_growth=True  # 随着进程逐渐增加显存占用，而不是一下占满
                              )
)
