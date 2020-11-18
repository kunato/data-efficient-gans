# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Main training script."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base

# ----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.


def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('DynamicRange'):
        x = tf.cast(x, tf.float32)
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            x = tf.where(tf.random_uniform(
                [tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    # Smooth crossfade between consecutive levels-of-detail.
    with tf.name_scope('FadeLOD'):
        s = tf.shape(x)
        y = tf.reshape(x, [-1, s[1], s[2] // 2, 2, s[3] // 2, 2])
        y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
        y = tf.tile(y, [1, 1, 1, 2, 1, 2])
        y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        x = tflib.lerp(x, y, lod - tf.floor(lod))
    # Upscale to match the expected input/output size of the networks.
    with tf.name_scope('UpscaleLOD'):
        s = tf.shape(x)
        factor = tf.cast(2 ** tf.floor(lod), tf.int32)
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

# ----------------------------------------------------------------------------
# Evaluate time-varying training parameters.


def training_schedule(
        cur_nimg,
        training_set,
        # Image resolution used at the beginning.
        lod_initial_resolution=None,
        # Thousands of real images to show before doubling the resolution.
        lod_training_kimg=600,
        # Thousands of real images to show when fading in new layers.
        lod_transition_kimg=600,
        minibatch_size_base=32,       # Global minibatch size.
        minibatch_size_dict={},       # Resolution-specific overrides.
        # Number of samples processed at a time by one GPU.
        minibatch_gpu_base=4,
        minibatch_gpu_dict={},       # Resolution-specific overrides.
        G_lrate_base=0.002,    # Learning rate for the generator.
        G_lrate_dict={},       # Resolution-specific overrides.
        D_lrate_base=0.002,    # Learning rate for the discriminator.
        D_lrate_dict={},       # Resolution-specific overrides.
        lrate_rampup_kimg=0,        # Duration of learning rate ramp-up.
        tick_kimg_base=4,        # Default interval of progress snapshots.
        tick_kimg_dict={8: 28, 16: 24, 32: 20, 64: 16, 128: 12, 256: 10, 512: 10, 1024: 10}):  # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    if lod_initial_resolution is None:
        s.lod = 0.0
    else:
        s.lod = training_set.resolution_log2
        s.lod -= np.floor(np.log2(lod_initial_resolution))
        s.lod -= phase_idx
        if lod_transition_kimg > 0:
            s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / \
                lod_transition_kimg
        s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch_size = minibatch_size_dict.get(
        s.resolution, minibatch_size_base)
    s.minibatch_gpu = minibatch_gpu_dict.get(s.resolution, minibatch_gpu_base)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

# ----------------------------------------------------------------------------
# Main training script.


def training_loop(
        G_args={},       # Options for generator network.
        D_args={},       # Options for discriminator network.
        G_opt_args={},       # Options for generator optimizer.
        D_opt_args={},       # Options for discriminator optimizer.
        loss_args={},       # Options for loss.
        dataset_args={},       # Options for dataset.load_dataset().
        sched_args={},       # Options for train.TrainingSchedule.
        grid_args={},       # Options for train.setup_snapshot_image_grid().
        metric_arg_list=[],       # Options for metrics.
        metric_args={},       # Options for MetricGroup.
        tf_config={},       # Options for tflib.init_tf().
        # Start of the exponential moving average. Default to the half-life period.
        ema_start_kimg=None,
        # Half-life of the exponential moving average of generator weights.
        G_ema_kimg=10,
        # Number of minibatches to run before adjusting training parameters.
        minibatch_repeats=4,
        lazy_regularization=False,    # Perform regularization as a separate training step?
        # How often the perform regularization for G? Ignored if lazy_regularization=False.
        G_reg_interval=4,
        # How often the perform regularization for D? Ignored if lazy_regularization=False.
        D_reg_interval=4,
        # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
        reset_opt_for_new_lod=True,
        # Total length of the training, measured in thousands of real images.
        total_kimg=25000,
        mirror_augment=False,    # Enable mirror augment?
        # Dynamic range used when feeding image data to the networks.
        drange_net=[-1, 1],
        # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
        image_snapshot_ticks=10,
        # How often to save network snapshots? None = only save 'networks-final.pkl'.
        network_snapshot_ticks=10,
        save_tf_graph=False,    # Include full TensorFlow computation graph in the tfevents file?
        save_weight_histograms=False,    # Include weight histograms in the tfevents file?
        # Network pickle to resume training from, None = train from scratch.
        resume_pkl=None,
        # Assumed training progress at the beginning. Affects reporting and training schedule.
        resume_kimg=0.0,
        # Assumed wallclock time at the beginning. Affects reporting.
        resume_time=0.0,
        resume_with_new_nets=False):   # Construct new networks according to G_args and D_args before resuming training?

    # resume_pkl = '/root/data-efficient-gans/DiffAugment-stylegan2/results/00002-DiffAugment-stylegan2-data-1024-batch16-4gpu-fmap8196-color-translation-cutout/network-snapshot-000401.pkl'
    # resume_kimg = 400.0
    print(f"Resume from {resume_pkl}@{resume_kimg}")
    if ema_start_kimg is None:
        ema_start_kimg = G_ema_kimg

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Load training set.
    training_set = dataset.load_dataset(verbose=True, **dataset_args)
    grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(
        training_set, **grid_args)
    misc.save_image_grid(grid_reals, dnnlib.make_run_dir_path(
        'reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)

    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            G = tflib.Network(
                'G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network(
                'D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
        if resume_pkl is not None:
            resume_networks = misc.load_pkl(resume_pkl)
            rG, rD, rGs = resume_networks
            if resume_with_new_nets:
                G.copy_vars_from(rG)
                D.copy_vars_from(rD)
                Gs.copy_vars_from(rGs)
            else:
                G, D, Gs = rG, rD, rGs

    # Print layers and generate initial image snapshot.
    G.print_layers()
    D.print_layers()
    sched = training_schedule(cur_nimg=total_kimg *
                              1000, training_set=training_set, **sched_args)
    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
    grid_fakes = Gs.run(grid_latents, grid_labels,
                        is_validation=True, minibatch_size=sched.minibatch_gpu)
    misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path(
        'fakes_init.png'), drange=drange_net, grid_size=grid_size)

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])
        G_lrate_in = tf.placeholder(tf.float32, name='G_lrate_in', shape=[])
        D_lrate_in = tf.placeholder(tf.float32, name='D_lrate_in', shape=[])
        minibatch_size_in = tf.placeholder(
            tf.int32, name='minibatch_size_in', shape=[])
        minibatch_gpu_in = tf.placeholder(
            tf.int32, name='minibatch_gpu_in', shape=[])
        run_D_reg_in = tf.placeholder(tf.bool, name='run_D_reg', shape=[])
        minibatch_multiplier = minibatch_size_in // (
            minibatch_gpu_in * num_gpus)
        Gs_beta_mul_in = tf.placeholder(
            tf.float32, name='Gs_beta_in', shape=[])
        Gs_beta = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32),
                                G_ema_kimg * 1000.0) if G_ema_kimg > 0.0 else 0.0

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    D_opt_args = dict(D_opt_args)
    G_opt_args['learning_rate'] = G_lrate_in
    D_opt_args['learning_rate'] = D_lrate_in
    for args in [G_opt_args, D_opt_args]:
        args['minibatch_multiplier'] = minibatch_multiplier
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)

    # Build training graph for each GPU.
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            with tf.name_scope('DataFetch'):
                reals_read, labels_read = training_set.get_minibatch_tf()
                reals_read = process_reals(
                    reals_read, lod_in, mirror_augment, training_set.dynamic_range, drange_net)

            # Create GPU-specific shadow copies of G and D.
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')

            # Evaluate loss functions.
            lod_assign_ops = []
            if 'lod' in G_gpu.vars:
                lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
            if 'lod' in D_gpu.vars:
                lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
            with tf.control_dependencies(lod_assign_ops):
                with tf.name_scope('loss'):
                    G_loss, D_loss, D_reg = dnnlib.util.call_func_by_name(
                        G=G_gpu, D=D_gpu, training_set=training_set, minibatch_size=minibatch_gpu_in, reals=reals_read, real_labels=labels_read, **loss_args)

            # Register gradients.
            if not lazy_regularization:
                if D_reg is not None:
                    D_loss += D_reg
            else:
                if D_reg is not None:
                    D_loss = tf.cond(run_D_reg_in, lambda: D_loss +
                                     D_reg * D_reg_interval, lambda: D_loss)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)

    # Setup training ops.
    Gs_update_op = Gs.setup_as_moving_average_of(
        G, beta=Gs_beta * Gs_beta_mul_in)
    with tf.control_dependencies([Gs_update_op]):
        G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    # Finalize graph.
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms()
        D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list, **metric_args)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop():
            break

        # Choose training parameters and configure training ops.
        sched = training_schedule(
            cur_nimg=cur_nimg, training_set=training_set, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state()
                D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        feed_dict = {
            lod_in: sched.lod,
            G_lrate_in: sched.G_lrate,
            D_lrate_in: sched.D_lrate,
            minibatch_size_in: sched.minibatch_size,
            minibatch_gpu_in: sched.minibatch_gpu,
            Gs_beta_mul_in: 1 if cur_nimg >= ema_start_kimg * 1000 else 0,
        }
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size,
                           sched.minibatch_gpu * num_gpus)
            run_D_reg = (lazy_regularization and running_mb_counter %
                         D_reg_interval == 0)
            feed_dict[run_D_reg_in] = run_D_reg
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            for _ in rounds:
                tflib.run(G_train_op, feed_dict)
                tflib.run(D_train_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch_size),
                dnnlib.util.format_time(autosummary(
                    'Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                grid_fakes = Gs.run(
                    grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch_gpu)
                misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path(
                    'fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path(
                    'network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)
                metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(),
                            num_gpus=num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % sched.lod,
                                           cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # Save final snapshot.
    misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()

# ----------------------------------------------------------------------------
