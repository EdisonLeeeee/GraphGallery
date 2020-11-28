import torch
import numpy as np
import tensorflow as tf


def train_step_tf(model, sequence, device):
    model.reset_metrics()

    with tf.device(device):
        for inputs, labels in sequence:
            results = model.train_on_batch(x=inputs,
                                           y=labels,
                                           reset_metrics=False)

    return dict(zip(model.metrics_names, results))


# def train_step_tf(model, sequence, device):
#     model.reset_metrics()
#     loss_fn = model.loss
#     metric = model.metrics[0]
#     optimizer = model.optimizer
#     model.reset_metrics()
#     metric.reset_states()

#     loss = 0.
#     with tf.GradientTape() as tape:
#         for inputs, labels in sequence:
#             output = model(inputs, training=True)
#             loss += loss_fn(labels, output)
#             metric.update_state(labels, output)

#     grad = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grad, model.trainable_variables))

#     return loss, metric.result()


def train_step_torch(model, sequence):
    model.train()
    optimizer = model.optimizer
    loss_fn = model.loss
    metrics = model.metrics
    model.reset_metrics()

    loss = torch.tensor(0.)
    for inputs, labels in sequence:
        optimizer.zero_grad()
        output = model(inputs)
        _loss = loss_fn(output, labels)
        _loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss += _loss.data
            for metric in metrics:
                metric.update_state(labels, output)

    results = [loss.detach().item()
               ] + [metric.result().item() for metric in metrics]
    return dict(zip(model.metrics_names, results))


def test_step_tf(model, sequence, device):
    model.reset_metrics()

    with tf.device(device):
        for inputs, labels in sequence:
            results = model.test_on_batch(x=inputs,
                                          y=labels,
                                          reset_metrics=False)

    return dict(zip(model.metrics_names, results))


# def test_step_tf(model, sequence, device):
#     model.reset_metrics()
#     loss_fn = model.loss
#     metric = model.metrics[0]
#     optimizer = model.optimizer
#     model.reset_metrics()

#     loss = 0.
#     for inputs, labels in sequence:
#         output = model(inputs, training=False)
#         loss += loss_fn(labels, output)
#         metric.update_state(labels, output)

#     return loss, metric.result()


@torch.no_grad()
def test_step_torch(model, sequence):
    model.eval()
    optimizer = model.optimizer
    loss_fn = model.loss
    metrics = model.metrics
    model.reset_metrics()
    loss = torch.tensor(0.)

    for inputs, labels in sequence:
        output = model(inputs)
        _loss = loss_fn(output, labels)
        loss += _loss.data
        for metric in metrics:
            metric.update_state(labels, output)

    results = [loss.detach().item()
               ] + [metric.result().item() for metric in metrics]
    return dict(zip(model.metrics_names, results))


def predict_step_tf(model, sequence, device):
    logits = []
    with tf.device(device):
        for inputs, *_ in sequence:
            logit = model.predict_on_batch(x=inputs)
            if tf.is_tensor(logit):
                logit = logit.numpy()
            logits.append(logit)

    if len(logits) > 1:
        logits = np.vstack(logits)
    else:
        logits = logits[0]
    return logits


# def predict_step_tf(model, sequence, device):
#     logits = []
#     with tf.device(device):
#         for inputs, *_ in sequence:
#             logit = model(inputs, training=False)
#             logits.append(logit)

#     if len(logits) > 1:
#         logits = tf.concat(logits, axis=0)
#     else:
#         logits = logits[0]

#     return logits.numpy()


@torch.no_grad()
def predict_step_torch(model, sequence):
    model.eval()
    logits = []

    for inputs, _ in sequence:
        logit = model(inputs)
        logits.append(logit)

    if len(logits) > 1:
        logits = torch.cat(logits)
    else:
        logits, = logits

    return logits.detach().cpu().numpy()
