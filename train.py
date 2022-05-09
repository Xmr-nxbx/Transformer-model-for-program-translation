import os
import time

import numpy as np
import tensorflow as tf

from util import vocab, preprocessing

config = {
    "hidden_size": 768,
    "head_num": 12,
    "layer_num": 12,
    "batch_size": 8,
    "epoch_num": 2,
    "input_length": 400,
    "output_length": 400,
    "dropout": .1,
    "warmup_step": 0,
    "learning_rate": 1e-4,

    "swap datasets by dictionary order": False,
    "gradient_accumulation": 1,
    "train": True,
    "test_batch": 1,
    "dataset_path": "./data",
    "model_name": "",
}


def init():
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_logical_device_configuration(
    #     gpus[0],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=3000),
    #      tf.config.LogicalDeviceConfiguration(memory_limit=3000)])
    config["strategy"] = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    print('Number of devices: %d' % config["strategy"].num_replicas_in_sync)
    config["batch_size"] *= config["strategy"].num_replicas_in_sync

    config["model_name"] = "Transformer_%d_%d_%d" % (
        config["hidden_size"], config["head_num"], config["layer_num"])
    if config["swap datasets by dictionary order"]:
        config["model_name"] += "_Swap"

    tf.random.set_seed(config.__sizeof__())
    np.random.seed(config.__sizeof__())

    get_dataset_path()


def get_dataset_path():
    dataset_list = os.listdir(config["dataset_path"])
    select = 0
    if len(dataset_list) == 0:
        exit(0)
    elif len(dataset_list) > 1:
        for i, each in enumerate(dataset_list):
            print("%d -> %s" % (i, each))
        select = input("select dataset by Index:\n")
        assert select.isdigit() and len(dataset_list) > int(select) >= 0
        select = int(select)
    else:
        select = 0
    config["dataset_path"] = os.path.join(config["dataset_path"], dataset_list[select])


def get_token2id():
    return vocab.vocab_token2id(os.path.join(config["dataset_path"], "vocab.json"))


def get_dataset(pad, first_id, last_id):
    import json
    file_list = os.listdir(config["dataset_path"])
    keys = []
    for each in file_list:
        if os.path.isfile(os.path.join(config["dataset_path"], each)) and not each.endswith("json"):
            keys.append(each)
    keys.sort(reverse=config["swap datasets by dictionary order"])
    print("Translate from key1(%s) to key2(%s)" % (keys[0], keys[1]))

    with open(os.path.join(config["dataset_path"], "data/train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(config["dataset_path"], "data/valid.json"), "r", encoding="utf-8") as f:
        valid_data = json.load(f)
    with open(os.path.join(config["dataset_path"], "data/test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)
    dataset_buffer = [train_data, valid_data, test_data]

    type = {"x": tf.int32, "y": tf.int32}
    shape = {"x": [None], "y": [None]}

    def generator(index):
        for each in dataset_buffer[index]:
            yield {"x": each[keys[0]][:config["input_length"]],
                   "y": (each[keys[1]] + [last_id])[:config["output_length"]]}

    def preprocess(data: dict):
        x_mask, memory_mask = preprocessing.get_transformer_encoder_data(data["x"], pad)
        y_input, y_input_mask, y_mask = preprocessing.get_transformer_decoder_data(data["y"], pad, first_id)

        encoder_data = {"x": data["x"], "x_mask": x_mask}
        decoder_data = {"y_input": y_input, "y_input_mask": y_input_mask, "y": data["y"], "y_mask": y_mask,
                        "memory_mask": memory_mask}
        return encoder_data, decoder_data

    train_dataset = tf.data.Dataset.from_generator(generator, args=(0,), output_types=type) \
        .prefetch(config["batch_size"]) \
        .padded_batch(config["batch_size"], shape, padding_values=pad) \
        .shuffle(4 * config["batch_size"]) \
        .map(preprocess, tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_generator(generator, args=(1,), output_types=type) \
        .prefetch(config["batch_size"]) \
        .padded_batch(config["batch_size"], shape, padding_values=pad) \
        .map(preprocess, tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(generator, args=(2,), output_types=type).prefetch(8) \
        .prefetch(config["test_batch"] * 4) \
        .padded_batch(config["test_batch"], shape, padding_values=pad) \
        .map(preprocess)

    return train_dataset, valid_dataset, test_dataset


def train():
    from network.TransformerUnit import Transformer, CustomSchedule
    from util.gradient import GradientAccumulator
    print("==========train==========")

    token2id = get_token2id()
    pad = token2id["<pad>"]
    first_id = token2id["<s>"]
    last_id = token2id["</s>"]
    config["vocab_size"] = len(token2id)

    config["model_path"] = os.path.join(config["dataset_path"], "model")
    config["model_path"] = os.path.join(config["model_path"], config["model_name"])
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])

    train_dataset, valid_dataset, _ = get_dataset(pad, first_id, last_id)

    train_db = config["strategy"].experimental_distribute_dataset(train_dataset)
    valid_db = config["strategy"].experimental_distribute_dataset(valid_dataset)

    with config["strategy"].scope():
        model = Transformer(config["vocab_size"], config["input_length"], config["output_length"],
                            config["hidden_size"], config["head_num"], 4 * config["hidden_size"],
                            config["layer_num"], config["dropout"])
        if config["warmup_step"] == 0:
            optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        else:
            lr_sechedule = CustomSchedule(config["hidden_size"], config["warmup_step"])
            optimizer = tf.keras.optimizers.Adam(lr_sechedule, .9, .98, 1e-9)
        now_epoch = tf.Variable(0)

        # init checkpoint and variables
        checkpoint = tf.train.Checkpoint(epoch=now_epoch, model=model, optimizer=optimizer)
        manager = tf.train.CheckpointManager(checkpoint, config["model_path"], checkpoint_name=config["model_name"],
                                             max_to_keep=3)
        if tf.train.latest_checkpoint(config["model_path"]):
            print('restore from checkpoint')
            checkpoint.restore(tf.train.latest_checkpoint(config["model_path"])).expect_partial()
            print('Now Epoch / All Epoch: %d / %d' % (now_epoch.numpy().tolist(), config["epoch_num"]))

        accumulator = GradientAccumulator()

    if config["gradient_accumulation"] < 1:
        config["gradient_accumulation"] = 1

    def apply_on_replica():
        optimizer.apply_gradients(zip(accumulator.gradients, model.trainable_variables))

    @tf.function
    def check_for_applying_gradient(require=config["gradient_accumulation"]):
        if accumulator.step >= require:
            config["strategy"].run(apply_on_replica)
            accumulator.reset()

    def train_step(encoder_data: dict, decoder_data: dict):
        with tf.GradientTape() as tape:
            logits, _ = model(encoder_data["x"], decoder_data["y_input"],
                              encoder_data["x_mask"], decoder_data["y_input_mask"],
                              decoder_data["memory_mask"], True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(decoder_data["y"], logits, True)
            y_mask = 1. - tf.cast(decoder_data["y_mask"], tf.float32)
            loss = tf.reduce_sum(loss * y_mask) / tf.reduce_sum(y_mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 4.)
        accumulator(gradients)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def valid_step(encoder_data: dict, decoder_data: dict):
        logits, weight = model(encoder_data["x"], decoder_data["y_input"],
                               encoder_data["x_mask"], decoder_data["y_input_mask"],
                               decoder_data["memory_mask"], False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(decoder_data["y"], logits, True)
        y_mask = 1. - tf.cast(decoder_data["y_mask"], tf.float32)
        loss = tf.reduce_sum(loss * y_mask) / tf.reduce_sum(y_mask)
        return loss, weight

    @tf.function(experimental_relax_shapes=True)
    def distributed_train_step(encoder_data, decoder_data):
        losses = config["strategy"].run(train_step, args=(encoder_data, decoder_data))
        if config["strategy"].num_replicas_in_sync > 1:
            losses = config["strategy"].reduce(tf.distribute.ReduceOp.SUM, losses, axis=None) / \
                     config["strategy"].num_replicas_in_sync
        return losses

    @tf.function(experimental_relax_shapes=True)
    def distributed_valid_step(encoder_data, decoder_data):
        losses, weights = config["strategy"].run(valid_step, args=(encoder_data, decoder_data))
        if config["strategy"].num_replicas_in_sync > 1:
            losses = config["strategy"].reduce(tf.distribute.ReduceOp.SUM, losses, axis=None) / \
                     config["strategy"].num_replicas_in_sync
            weights = [weights[i].values[0] for i in range(len(weights))]
        return losses, weights

    start_epoch = now_epoch.numpy()
    training_start_time = time.time()
    for epoch in range(start_epoch, config["epoch_num"]):

        total_loss = 0.0
        num_batches = 0
        total_start_time = time.time()
        start_time = time.time()
        for x_data, y_data in train_db:
            loss = distributed_train_step(x_data, y_data)
            total_loss += loss.numpy()
            num_batches += 1
            if epoch == start_epoch and num_batches == 1:
                model.summary()
            check_for_applying_gradient()
            if num_batches % 50 == 0:
                print("Epoch %d, batch num: %d, time: %ds, loss: %.4f" %
                      (epoch, num_batches, time.time() - start_time, loss))
                start_time = time.time()

        check_for_applying_gradient(1)
        train_loss = total_loss / num_batches
        print("Epoch %d, time: %ds, training loss: %.4f" % (epoch, time.time() - total_start_time, train_loss))

        total_loss = 0.0
        num_batches = 0
        total_start_time = time.time()
        for x_data, y_data in valid_db:
            loss, weight = distributed_valid_step(x_data, y_data)
            total_loss += loss.numpy()
            num_batches += 1

        valid_loss = total_loss / num_batches
        with open(os.path.join(config["model_path"], "loss"), "a", encoding="utf-8") as f:
            f.write("%f\t%f\n" % (train_loss, valid_loss))
        print("Epoch %d, time: %ds, valid loss: %.4f\n" % (epoch, time.time() - total_start_time, valid_loss))

        now_epoch.assign_add(1)
        manager.save(checkpoint_number=epoch + 1)

    if start_epoch != config["epoch_num"]:
        print("training time: %ds" % (time.time() - training_start_time))
    print("==============================\n\n")


def test():
    from network.TransformerUnit import Transformer, CustomSchedule
    print("==========test==========")

    token2id = get_token2id()
    id2token = {v: k for k, v in token2id.items()}
    pad = token2id["<pad>"]
    first_id = token2id["<s>"]
    last_id = token2id["</s>"]
    config["vocab_size"] = len(token2id)

    config["model_path"] = os.path.join(config["dataset_path"], "model")
    config["model_path"] = os.path.join(config["model_path"], config["model_name"])
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])

    _, _, test_dataset = get_dataset(pad, first_id, last_id)

    model = Transformer(config["vocab_size"], config["input_length"], config["output_length"],
                        config["hidden_size"], config["head_num"], 4 * config["hidden_size"],
                        config["layer_num"], config["dropout"])
    if config["warmup_step"] == 0:
        optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
    else:
        lr_sechedule = CustomSchedule(config["hidden_size"], config["warmup_step"])
        optimizer = tf.keras.optimizers.Adam(lr_sechedule, .9, .98, 1e-9)
    now_epoch = tf.Variable(0)

    # init checkpoint and variables
    checkpoint = tf.train.Checkpoint(epoch=now_epoch, model=model, optimizer=optimizer)
    if tf.train.latest_checkpoint(config["model_path"]):
        print('restore from checkpoint')
        checkpoint.restore(tf.train.latest_checkpoint(config["model_path"])).expect_partial()
    else:
        print('no checkpoint exists')
        exit(0)

    @tf.function(experimental_relax_shapes=True)
    def run(x, y_input, x_mask, memory_mask):
        logtis, _ = model(x, y_input, x_mask, None, memory_mask, False)
        return logtis

    def test_step(encoder_data, decoder_data):
        decoder_input = tf.constant([[first_id]] * config["test_batch"], tf.int32)
        decoder_output = tf.constant([[]] * config["test_batch"], tf.int32)
        output_flag = tf.zeros([config["test_batch"]], tf.int32)
        while tf.shape(decoder_output)[-1] < config["output_length"]:

            logits = run(encoder_data["x"], decoder_input, encoder_data["x_mask"], decoder_data["memory_mask"])
            now = tf.argmax(logits, axis=-1, output_type=tf.int32)[:, -1:]

            end_place = tf.cast(now[:, 0] == last_id, tf.int32)
            output_flag = tf.maximum(output_flag, end_place)
            if tf.reduce_sum(output_flag) == config["test_batch"]:
                break

            decoder_input = tf.concat([decoder_input, now], -1)
            decoder_output = tf.concat([decoder_output, now], -1)
        return decoder_output

    def transformation(outputs):
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(10)

        def func(index):
            token_seq = []
            for token_id in outputs[index]:
                token = id2token[token_id]
                if token == "</s>":
                    break
                token_seq.append(token)
            token_text = "".join(token_seq).replace("▁", " ").replace("$$", "")
            token_text = token_text.replace("<pad>", "").replace("</s>", "")
            outputs[index] = token_text.strip()

        pool.map(func, list(range(len(outputs))))
        pool.close()
        pool.join()

    start_time = time.time()
    nums = 0
    outputs = []
    for x_data, y_data in iter(test_dataset):
        output = test_step(x_data, y_data).numpy()  # [b, seq]
        outputs.extend(output)
        nums += 1
        if nums % 5 == 0:
            print("test nums: %d, time: %ds" % (nums, time.time() - start_time))
            start_time = time.time()

    transformation(outputs)

    with open(os.path.join(config["model_path"], "out"), "w", encoding="utf-8") as f:
        f.write("\n".join(outputs) + "\n")

    print("==============================")


if __name__ == '__main__':
    init()
    if config["train"]:
        train()
    test()
