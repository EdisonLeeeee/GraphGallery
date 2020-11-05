import torch
import numpy as np
import tensorflow as tf


def train_step_tf(model, sequence, device):
    """
    Train the model on the model.

    Args:
        model: (todo): write your description
        sequence: (todo): write your description
        device: (todo): write your description
    """
    model.reset_metrics()

    with tf.device(device):
        for inputs, labels in sequence:
            metrics = model.train_on_batch(
                x=inputs, y=labels, reset_metrics=False)

    return metrics

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
    """
    Train the model

    Args:
        model: (todo): write your description
        sequence: (todo): write your description
    """
    model.train()
    optimizer = model.optimizer
    loss_fn = model.loss_fn

    accuracy = 0.
    loss = 0.
    n_inputs = 0

    for inputs, labels in sequence:
        optimizer.zero_grad()
        output = model(inputs)
        _loss = loss_fn(output, labels)
        _loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss += _loss.data
            accuracy += (output.argmax(1) == labels).float().sum()
            n_inputs += labels.size(0)

    return loss.detach().item(), (accuracy / n_inputs).detach().item()


def test_step_tf(model, sequence, device):
    """
    Perform a single tf.

    Args:
        model: (todo): write your description
        sequence: (todo): write your description
        device: (todo): write your description
    """
    model.reset_metrics()

    with tf.device(device):
        for inputs, labels in sequence:
            metrics = model.test_on_batch(
                x=inputs, y=labels, reset_metrics=False)

    return metrics

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
    """
    Evaluate loss.

    Args:
        model: (todo): write your description
        sequence: (todo): write your description
    """
    model.eval()
    loss_fn = model.loss_fn
    accuracy = 0.
    loss = 0.
    n_inputs = 0

    for inputs, labels in sequence:
        output = model(inputs)
        _loss = loss_fn(output, labels)
        loss += _loss.data
        n_inputs += labels.size(0)
        accuracy += (output.argmax(1) == labels).float().sum()

    return loss.detach().item(), (accuracy / n_inputs).detach().item()


def predict_step_tf(model, sequence, device):
    """
    Predict the model on the given model.

    Args:
        model: (todo): write your description
        sequence: (todo): write your description
        device: (todo): write your description
    """
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
    """
    Predict the logits the model.

    Args:
        model: (todo): write your description
        sequence: (todo): write your description
    """
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
